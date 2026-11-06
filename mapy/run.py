import sys, os, json, math, random, time, re, datetime, threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set SUMO_HOME")

import traci
import traci.constants as tc
import sumolib

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QTextEdit, QCheckBox,
    QGroupBox, QFormLayout, QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QHeaderView, QFileDialog, QTimeEdit, QStackedWidget
)
from PySide6.QtCore import QThread, Signal, Qt, QPointF, QTime
from PySide6.QtGui import (
    QPainter, QPainterPath, QPen, QBrush, QColor, QPixmap, QPolygonF, QFont, QTransform
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(SCRIPT_DIR, "GliwiceMid.net.xml")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "tls_settings")
STATE_DIR = os.path.join(SCRIPT_DIR, "tmp_states")

BIN_SECONDS = 15 * 60      # raporty co 15 minut

SIM_SEED = "42"            # ten sam seed dla żywej symulacji i bazy
EVAL_SEEDS = (11, 22)      # ziarna do oceny kandydatów, wynik = średnia
VALIDATION_SEEDS = (33, 44)  # świeże ziarna do potwierdzenia zwycięzcy
MIN_GAIN = 0.02            # zmiana musi być lepsza od kontroli o co najmniej 2%
MAX_DEVIATION = 15.0       # maks. odchylenie fazy od oryginału [s]
MIN_SCORE_TO_TUNE = 100.0  # poniżej tylu sekund stania w oknie nie ma czego poprawiać

# tempo podglądu: ile sekund symulacji przypada na 1 sekundę rzeczywistą (0 = tak szybko, jak się da)
SPEED_OPTIONS = [("maks.", 0), ("×10", 10), ("×30", 30), ("×60", 60), ("×120", 120), ("×300", 300)]

# sygnalizacja akomodacyjna (wbudowany w SUMO typ "actuated")
ACT_PROGRAM_ID = "act"
ACTUATED_ADD_FILE = os.path.join(SCRIPT_DIR, "actuated_tls.add.xml")

# natężenie ruchu z Excela/CSV i okno godzin (wymagane we wszystkich trybach)
WINDOW_ROUTE_FILE = os.path.join(SCRIPT_DIR, "GliwiceMid_window.rou.xml")
OD_POOL_FILE = os.path.join(SCRIPT_DIR, "od_pool.trips.xml")


# ============================================================
# --- SUMO / TLS helpers (przyjmują jawne połączenie `conn`) ---
# ============================================================

def is_yellow_phase(state):
    return "y" in state.lower()


def classify_phase(state):
    s = state.lower()
    if "g" in s:
        return "green"
    if s.count("r") == len(s):
        return "red"
    return "mixed"


def static_car_relevant_tls(net):
    """Sygnalizacje sterujące ruchem samochodowym — wyznaczone z pliku sieci (bez TraCI)."""
    ids = []
    tls_list = net.getTrafficLights() if hasattr(net, "getTrafficLights") else net.getTLSs()
    for tls in tls_list:
        edge_types = {e.getType() for e in tls.getEdges()}
        if any(not t.startswith("railway") for t in edge_types):
            ids.append(tls.getID())
    return ids


def build_tls_layout(tls_ids, log, conn, min_tunable=0.0):
    layout = {}
    offset = 0
    for tls_id in tls_ids:
        program = conn.trafficlight.getAllProgramLogics(tls_id)[0]
        # w programach akomodacyjnych (actuated itp.) o czasie decyduje minDur/maxDur, więc zmiana
        # `duration` nic by nie dała — takie sygnalizacje pomijamy
        is_static = getattr(program, "type", 0) in (0, "0", "static")
        tunable_indices = [i for i, ph in enumerate(program.phases)
                           if is_static and is_actuatable_phase(ph.state) and ph.duration >= min_tunable]
        layout[tls_id] = (offset, tunable_indices)
        offset += len(tunable_indices)
        if not is_static:
            log(f"{tls_id}: program typu {program.type} (nie stałoczasowy) — pomijam w optymalizacji")
        log(f"Phases for {tls_id}:")
        for i, phase in enumerate(program.phases):
            tag = "" if i in tunable_indices else " (stała)"
            log(f"  [{i}] state={phase.state} duration={phase.duration}{tag}")
    return layout, offset


def build_baseline_vector(layout, tls_ids, conn):
    dims = sum(len(indices) for _, indices in layout.values())
    baseline = np.zeros(dims)
    for tls_id in tls_ids:
        program = conn.trafficlight.getAllProgramLogics(tls_id)[0]
        start, tunable_indices = layout[tls_id]
        for offset, idx in enumerate(tunable_indices):
            baseline[start + offset] = program.phases[idx].duration
    return baseline


def apply_durations(layout, phase_durations, conn):
    for tls_id, (start, tunable_indices) in layout.items():
        if not tunable_indices:
            continue     # np. sygnalizacja akomodacyjna: nie ruszamy jej programu
        durations = phase_durations[start:start + len(tunable_indices)]
        program = conn.trafficlight.getAllProgramLogics(tls_id)[0]
        for idx, dur in zip(tunable_indices, durations):
            program.phases[idx].duration = round(float(dur), 1)
        conn.trafficlight.setProgramLogic(tls_id, program)


def start_sumo(cmd, prefix, attempts=4):
    """Uruchamia SUMO z unikalną etykietą połączenia (bez kolizji 'already active')."""
    last_error = None
    for attempt in range(attempts):
        label = f"{prefix}_{os.getpid()}_{time.time_ns()}"
        try:
            traci.start(cmd, port=sumolib.miscutils.getFreeSocketPort(), label=label)
            return traci.getConnection(label)
        except Exception as e:
            last_error = e
            time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"nie udało się uruchomić SUMO po {attempts} próbach: {last_error}")


def close_conn(conn):
    try:
        conn.close()
    except Exception:
        pass


def write_tls_settings(layout, phase_durations, out_dir, log, route_file, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    conn = start_sumo(["sumo", "-n", NET_FILE, "-r", route_file, "--no-step-log"], "settings")
    try:
        for tls_id, (start, tunable_indices) in layout.items():
            program = conn.trafficlight.getAllProgramLogics(tls_id)[0]
            durations = phase_durations[start:start + len(tunable_indices)]
            applied = dict(zip(tunable_indices, durations))
            phases_out = []
            for i, phase in enumerate(program.phases):
                duration = applied.get(i, phase.duration)
                phases_out.append({
                    "index": i, "state": phase.state,
                    "duration": round(float(duration), 1),
                    "yellow_fixed": is_yellow_phase(phase.state),
                })
            path = os.path.join(out_dir, f"{tls_id}.json")
            with open(path, "w") as f:
                data = {"tls_id": tls_id, "phases": phases_out}
                if meta:
                    data["scenario"] = meta
                json.dump(data, f, indent=2)
            log(f"Wrote settings for {tls_id} -> {path}")
    finally:
        close_conn(conn)


def is_actuatable_phase(state):
    """Faza zielona (z jakimś zielonym, bez żółtego) może być przedłużana/skracana przez detektory."""
    low = state.lower()
    return "g" in low and "y" not in low


def write_actuated_additional(tls_ids, conn, path, params, log):
    """Zapisuje plik additional z programem 'actuated' dla każdej sygnalizacji.
    Fazy i stany są kopiowane z istniejącego programu; fazy zielone dostają zakres minDur-maxDur."""
    lines = ["<additional>"]
    for tid in tls_ids:
        prog = conn.trafficlight.getAllProgramLogics(tid)[0]
        lines.append(f'    <tlLogic id="{tid}" type="actuated" programID="{ACT_PROGRAM_ID}" offset="0">')
        lines.append(f'        <param key="max-gap" value="{params["max_gap"]}"/>')
        lines.append(f'        <param key="detector-gap" value="{params["detector_gap"]}"/>')
        lines.append(f'        <param key="passing-time" value="{params["passing_time"]}"/>')
        n_act = 0
        for ph in prog.phases:
            d = float(ph.duration)
            if is_actuatable_phase(ph.state):
                mn = min(d, max(params["floor"], params["min_pct"] / 100.0 * d))
                mx = max(d, params["max_pct"] / 100.0 * d)
                lines.append(f'        <phase duration="{d:.1f}" minDur="{mn:.1f}" maxDur="{mx:.1f}" state="{ph.state}"/>')
                n_act += 1
            else:
                lines.append(f'        <phase duration="{d:.1f}" state="{ph.state}"/>')
        lines.append("    </tlLogic>")
        log(f"  {tid}: {n_act} faz akomodacyjnych z {len(prog.phases)}")
    lines.append("</additional>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def format_time_of_day(seconds):
    h = int(seconds // 3600) % 24
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"


# ============================================================
# --- Subskrypcje TraCI, klatki podglądu, tempo ---
# ============================================================

class StopRequested(Exception):
    """Zgłaszany w wątku roboczym po anulowaniu."""


class RunControl:
    """Wstrzymywanie i anulowanie pracy wątku roboczego. GUI wywołuje pause/resume/cancel,
    a wątek roboczy w kilku miejscach wywołuje checkpoint()."""

    def __init__(self, running=None, cancelled=None):
        # zdarzenia z multiprocessing pozwalają wstrzymywać i anulować także procesy potomne
        if running is None:
            running = threading.Event()
            running.set()
        if cancelled is None:
            cancelled = threading.Event()
        self._running, self._cancelled = running, cancelled

    def shared_events(self):
        return self._running, self._cancelled

    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()

    def cancel(self):
        self._cancelled.set()
        self._running.set()        # budzi wątek, który czeka na wznowienie

    @property
    def paused(self):
        return not self._running.is_set()

    def checkpoint(self):
        """Blokuje na czas pauzy, a po anulowaniu zgłasza StopRequested.
        Zwraca True, jeśli była pauza (wtedy warto wyzerować licznik tempa podglądu)."""
        was_paused = False
        while not self._running.is_set():
            was_paused = True
            self._running.wait(0.1)
        if self._cancelled.is_set():
            raise StopRequested()
        return was_paused


_WORKER_CONTROL = None      # RunControl w procesie potomnym (ustawiany przez _init_worker)


def _init_worker(running, cancelled):
    global _WORKER_CONTROL
    _WORKER_CONTROL = RunControl(running, cancelled)


def make_pool(ctx, n_processes, control):
    """Pula procesów, której zadania reagują na pauzę i anulowanie z procesu głównego."""
    return ProcessPoolExecutor(max_workers=n_processes, mp_context=ctx,
                               initializer=_init_worker, initargs=control.shared_events())


class VehicleTracker:
    """Śledzi pozycje i prędkości pojazdów jednym zapytaniem na krok (subskrypcje TraCI)."""

    def __init__(self, conn):
        self.conn = conn
        conn.simulation.subscribe([tc.VAR_DEPARTED_VEHICLES_IDS])

    def update(self):
        """Wywołaj po simulationStep(). Zwraca {id: {VAR_POSITION: (x, y), VAR_SPEED: v}}."""
        departed = self.conn.simulation.getSubscriptionResults().get(tc.VAR_DEPARTED_VEHICLES_IDS, ())
        for vid in departed:
            self.conn.vehicle.subscribe(vid, [tc.VAR_POSITION, tc.VAR_SPEED])
        return self.conn.vehicle.getAllSubscriptionResults()


def build_frame(conn, results, tls_ids, sim_time):
    rows = []
    for data in results.values():
        pos = data.get(tc.VAR_POSITION)
        spd = data.get(tc.VAR_SPEED)
        if pos is None or spd is None:
            continue
        rows.append((pos[0], pos[1], spd))
    veh = np.array(rows, dtype=float).reshape(-1, 3)
    tls = {tid: conn.trafficlight.getRedYellowGreenState(tid) for tid in tls_ids}
    return {"t": sim_time, "veh": veh, "tls": tls}


class FramePacer:
    """Ogranicza liczbę klatek i (opcjonalnie) tempo symulacji względem czasu rzeczywistego."""

    def __init__(self, speed_factor, fps=10):
        self.speed = speed_factor
        self.min_dt = 1.0 / fps
        self.last_emit = 0.0
        self.reset(0.0)

    def reset(self, sim_time):
        self.wall0 = time.perf_counter()
        self.sim0 = sim_time

    def due(self):
        now = time.perf_counter()
        if now - self.last_emit >= self.min_dt:
            self.last_emit = now
            return True
        return False

    def throttle(self, sim_time):
        if self.speed <= 0:
            return
        ahead = (sim_time - self.sim0) / self.speed - (time.perf_counter() - self.wall0)
        if ahead > 0:
            time.sleep(min(ahead, 0.5))


# ============================================================
# --- Tryb zwykły: skończone symulacje ---
# ============================================================

def run_simulation(phase_durations=None, layout=None, route_file=None, end_time=None,
                   measure_from=0.0, control=None, seed=None):
    """Przebieg bez podglądu. Zwraca (suma czasu oczekiwania, śr. prędkość w m/s).
    end_time: koniec symulacji w s (brak = do dojazdu wszystkich pojazdów).
    measure_from: wcześniejsze sekundy (rozgrzewka) nie wchodzą do wyniku.
    control: RunControl do wstrzymywania/anulowania (sprawdzany co 10 kroków).
    seed: ziarno losowe SUMO (brak = domyślne)."""
    cmd = ["sumo", "-n", NET_FILE, "-r", route_file, "--no-step-log"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    conn = start_sumo(cmd, "run")
    total_wait = 0.0
    total_speed_samples = 0.0
    n_speed_samples = 0
    try:
        if phase_durations is not None and layout is not None:
            apply_durations(layout, phase_durations, conn)
        steps = 0
        while True:
            if control is not None and steps % 10 == 0:
                control.checkpoint()
            steps += 1
            if not end_time and conn.simulation.getMinExpectedNumber() <= 0:
                break
            conn.simulationStep()
            t = conn.simulation.getTime()
            if t > measure_from:
                for veh_id in conn.vehicle.getIDList():
                    total_wait += conn.vehicle.getWaitingTime(veh_id)
                    speed = conn.vehicle.getSpeed(veh_id)
                    if speed > 0.1:
                        total_speed_samples += speed
                        n_speed_samples += 1
            if end_time and t >= end_time:
                break
    finally:
        close_conn(conn)
    avg_speed = total_speed_samples / n_speed_samples if n_speed_samples else 0.0
    return total_wait, avg_speed


def replay_simulation(phase_durations, layout, tls_ids, frame_cb, speed_factor,
                      end_time=None, route_file=None, clock_offset=0.0, control=None, seed=None):
    """Odtwarza przebieg w podglądzie aplikacji (klatki przekazywane przez frame_cb).
    clock_offset: przesunięcie, dzięki któremu podgląd pokazuje godzinę z okna, a nie czas symulacji."""
    cmd = ["sumo", "-n", NET_FILE, "-r", route_file, "--no-step-log"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    conn = start_sumo(cmd, "replay")
    try:
        if phase_durations is not None and layout is not None:
            apply_durations(layout, phase_durations, conn)
        tracker = VehicleTracker(conn)
        pacer = FramePacer(speed_factor)
        while True:
            if control is not None and control.checkpoint():
                pacer.reset(conn.simulation.getTime())
            if not end_time and conn.simulation.getMinExpectedNumber() <= 0:
                break
            conn.simulationStep()
            t = conn.simulation.getTime()
            results = tracker.update()
            if pacer.due():
                frame_cb(build_frame(conn, results, tls_ids, t + clock_offset))
            pacer.throttle(t)
            if end_time and t >= end_time:
                break
    finally:
        close_conn(conn)


# ============================================================
# --- Algorytmy rojowe (interfejs wsadowy) ---
# evaluate_batch(lista_wektorów) -> lista_wyników
# bounds = (lo, hi), każde skalarem albo tablicą o długości dims
# ============================================================

def sparse_perturb(vector, noise, groups, dims):
    """Perturbuje tylko 1-3 losowe skrzyżowania naraz (groups = zakresy współrzędnych skrzyżowań).
    Zmiana wszystkich faz naraz prawie zawsze pogarsza wynik, a poprawa na jednym skrzyżowaniu ginie
    w szumie pozostałych."""
    out = np.array(vector, dtype=float)
    if not groups:
        return out + np.random.uniform(-noise, noise, size=dims)
    k = np.random.randint(1, min(3, len(groups)) + 1)
    for gi in np.random.choice(len(groups), size=k, replace=False):
        a, b = groups[gi]
        out[a:b] += np.random.uniform(-noise, noise, size=b - a)
    return out


def seed_population(n_individuals, dims, bounds, start_vector, seed_noise, n_random, groups=None):
    lo, hi = bounds
    population = []
    for i in range(n_individuals):
        if i < n_individuals - n_random:
            ind = sparse_perturb(start_vector, seed_noise, groups, dims)
        else:
            ind = np.random.uniform(lo, hi, size=dims)
        population.append(np.clip(ind, lo, hi))
    return np.array(population)


def should_stop(best_score, previous_score, same_count, patience=4):
    """Zwraca (stop, same_count, previous_score). patience=0 wyłącza wczesne zatrzymanie."""
    if patience and same_count > patience:
        return True, same_count, previous_score
    if previous_score == best_score:
        same_count += 1
    else:
        same_count = 1
        previous_score = best_score
    return False, same_count, previous_score


def run_pso(evaluate_batch, dims, bounds, start_vector, start_score, cfg, on_iteration):
    lo, hi = bounds
    n = cfg["n"]
    # Populacja startowa zmienia tylko kilka skrzyżowań naraz (sparse_perturb), a rój przyciąga najlepszy
    # znany wynik, łącznie z ustawieniami bazowymi: gdy baza jest już dobra, rój szuka wokół niej.
    particles = seed_population(n, dims, bounds, start_vector, cfg["noise"], cfg["n_random"], cfg.get("groups"))
    velocities = np.zeros((n, dims))
    personal_best = particles.copy()
    personal_best_score = np.full(n, np.inf)
    global_best, global_best_score = start_vector.copy(), start_score
    previous_score, same_count = np.inf, 1

    for it in range(cfg["iterations"]):
        scores = evaluate_batch(list(particles))
        for i, score in enumerate(scores):
            if score < personal_best_score[i]:
                personal_best[i], personal_best_score[i] = particles[i].copy(), score
            if score < global_best_score:
                global_best, global_best_score = particles[i].copy(), score

        r1 = np.random.rand(n, dims)
        r2 = np.random.rand(n, dims)
        velocities = (0.5 * velocities
                      + 1.5 * r1 * (personal_best - particles)
                      + 1.5 * r2 * (global_best - particles))
        particles = np.clip(particles + velocities, lo, hi)

        stop, same_count, previous_score = should_stop(global_best_score, previous_score, same_count,
                                                       cfg.get("patience", 4))
        on_iteration(it + 1, cfg["iterations"], global_best_score, global_best)
        if stop:
            break
    return global_best, global_best_score


def run_abc(evaluate_batch, dims, bounds, start_vector, start_score, cfg, on_iteration):
    lo, hi = bounds
    n = cfg["n"]
    limit = 5
    food = seed_population(n, dims, bounds, start_vector, cfg["noise"], cfg["n_random"], cfg.get("groups"))
    scores = np.array(evaluate_batch(list(food)), dtype=float)
    trials = np.zeros(n, dtype=int)
    best_idx = int(np.argmin(scores))
    if scores[best_idx] < start_score:
        global_best, global_best_score = food[best_idx].copy(), scores[best_idx]
    else:
        global_best, global_best_score = start_vector.copy(), start_score
    previous_score, same_count = np.inf, 1

    for it in range(cfg["iterations"]):
        for phase in ("employed", "onlooker"):
            if phase == "employed":
                indices = np.arange(n)
            else:
                # prawdopodobieństwa znormalizowane do rozrzutu wyników; surowe różnice rzędu
                # setek sekund zamieniłyby wybór pszczół obserwatorek w czysto zachłanny
                rel = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                fitness = 1.0 / (1.0 + 5.0 * rel)
                indices = np.random.choice(n, size=n, p=fitness / fitness.sum())

            candidates = []
            for i in indices:
                k = np.random.choice([j for j in range(n) if j != i])
                phi = np.random.uniform(-1, 1, size=dims)
                candidates.append(np.clip(food[i] + phi * (food[i] - food[k]), lo, hi))
            for i, cand, cs in zip(indices, candidates, evaluate_batch(candidates)):
                if cs < scores[i]:
                    food[i], scores[i], trials[i] = cand, cs, 0
                else:
                    trials[i] += 1

        # zwiadowczynie: porzucone źródła zastępujemy nowymi, wokół punktu startowego
        exhausted = [i for i in range(n) if trials[i] > limit]
        if exhausted:
            fresh = [np.clip(sparse_perturb(start_vector, cfg["noise"], cfg.get("groups"), dims), lo, hi)
                     for _ in exhausted]
            for i, vec, s in zip(exhausted, fresh, evaluate_batch(fresh)):
                food[i], scores[i], trials[i] = vec, s, 0

        best_idx = int(np.argmin(scores))
        if scores[best_idx] < global_best_score:
            global_best, global_best_score = food[best_idx].copy(), scores[best_idx]

        stop, same_count, previous_score = should_stop(global_best_score, previous_score, same_count, cfg.get("patience", 4))
        on_iteration(it + 1, cfg["iterations"], global_best_score, global_best)
        if stop:
            break
    return global_best, global_best_score


def run_fa(evaluate_batch, dims, bounds, start_vector, start_score, cfg, on_iteration):
    lo, hi = bounds
    n = cfg["n"]
    alpha, beta0, gamma = 5.0, 1.0, 0.001
    fireflies = seed_population(n, dims, bounds, start_vector, cfg["noise"], cfg["n_random"], cfg.get("groups"))
    scores = np.array(evaluate_batch(list(fireflies)), dtype=float)
    best_idx = int(np.argmin(scores))
    if scores[best_idx] < start_score:
        global_best, global_best_score = fireflies[best_idx].copy(), scores[best_idx]
    else:
        global_best, global_best_score = start_vector.copy(), start_score
    previous_score, same_count = np.inf, 1

    for it in range(cfg["iterations"]):
        # wersja synchroniczna: ruchy liczone ze stanu z początku iteracji, potem jedna wspólna ocena
        old_pos, old_scores = fireflies.copy(), scores.copy()
        moved_idx, moved_pos = [], []
        for i in range(n):
            brighter = np.where(old_scores < old_scores[i])[0]
            if len(brighter) == 0:
                continue   # najjaśniejszy świetlik zostaje w miejscu
            pos = old_pos[i].copy()
            for j in brighter:
                r2 = np.sum((old_pos[i] - old_pos[j]) ** 2)
                pos += beta0 * np.exp(-gamma * r2) * (old_pos[j] - pos)
            pos += alpha * (np.random.rand(dims) - 0.5)
            moved_idx.append(i)
            moved_pos.append(np.clip(pos, lo, hi))
        if moved_pos:
            for i, vec, s in zip(moved_idx, moved_pos, evaluate_batch(moved_pos)):
                fireflies[i], scores[i] = vec, s

        best_idx = int(np.argmin(scores))
        if scores[best_idx] < global_best_score:
            global_best, global_best_score = fireflies[best_idx].copy(), scores[best_idx]
        alpha *= 0.97

        stop, same_count, previous_score = should_stop(global_best_score, previous_score, same_count, cfg.get("patience", 4))
        on_iteration(it + 1, cfg["iterations"], global_best_score, global_best)
        if stop:
            break
    return global_best, global_best_score


def run_aco(evaluate_batch, dims, bounds, start_vector, start_score, cfg, on_iteration):
    lo, hi = bounds
    n = cfg["n"]
    archive = seed_population(n, dims, bounds, start_vector, cfg["noise"], cfg["n_random"], cfg.get("groups"))
    scores = np.array(evaluate_batch(list(archive)), dtype=float)
    order = np.argsort(scores)
    archive, scores = archive[order], scores[order]
    if scores[0] < start_score:
        global_best, global_best_score = archive[0].copy(), scores[0]
    else:
        global_best, global_best_score = start_vector.copy(), start_score
    q, xi = 0.3, 0.6
    ranks = np.arange(1, n + 1)
    weights = (1.0 / (q * n * np.sqrt(2 * np.pi))) * np.exp(-((ranks - 1) ** 2) / (2 * q ** 2 * n ** 2))
    weights /= weights.sum()
    previous_score, same_count = np.inf, 1

    for it in range(cfg["iterations"]):
        new_solutions = []
        for _ in range(n):
            chosen = np.random.choice(n, p=weights)
            sigma = xi * np.sum(np.abs(archive - archive[chosen]), axis=0) / (n - 1 + 1e-9)
            new_solutions.append(np.clip(np.random.normal(archive[chosen], sigma + 1e-6), lo, hi))
        new_scores = np.array(evaluate_batch(new_solutions), dtype=float)

        combined = np.vstack([archive, new_solutions])
        combined_scores = np.concatenate([scores, new_scores])
        order = np.argsort(combined_scores)[:n]
        archive, scores = combined[order], combined_scores[order]

        if scores[0] < global_best_score:
            global_best, global_best_score = archive[0].copy(), scores[0]

        stop, same_count, previous_score = should_stop(global_best_score, previous_score, same_count, cfg.get("patience", 4))
        on_iteration(it + 1, cfg["iterations"], global_best_score, global_best)
        if stop:
            break
    return global_best, global_best_score


ALGORITHMS = {"PSO": run_pso, "ABC": run_abc, "FA": run_fa, "ACO": run_aco}


# ============================================================
# --- Natężenie ruchu z Excela/CSV i okno godzin ---
# ============================================================

def parse_time_cell(cell):
    """Komórka z godziną -> sekundy od północy (albo None).
    Obsługuje: czas z Excela, tekst 'GG:MM[:SS]' oraz liczby (0-1 = ułamek doby jak w Excelu, 1-24 = godziny)."""
    if cell is None or isinstance(cell, bool):
        return None
    if isinstance(cell, datetime.datetime):
        return cell.hour * 3600 + cell.minute * 60 + cell.second
    if isinstance(cell, datetime.time):
        return cell.hour * 3600 + cell.minute * 60 + cell.second
    if isinstance(cell, (int, float)):
        v = float(cell)
        if 0 <= v < 1:
            return int(round(v * 86400))
        if 1 <= v <= 24:
            return int(round(v * 3600))
        return None
    if isinstance(cell, str):
        s = cell.strip()
        m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
        if m:
            h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            return h * 3600 + mi * 60 + se if (h <= 24 and mi < 60 and se < 60) else None
        try:
            return parse_time_cell(float(s.replace(",", ".")))
        except ValueError:
            return None
    return None


def parse_number(cell):
    if cell is None or isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        v = float(cell)
    elif isinstance(cell, str):
        try:
            v = float(cell.strip().replace(" ", "").replace(",", "."))
        except ValueError:
            return None
    else:
        return None
    return v if (math.isfinite(v) and v >= 0) else None


def load_demand_file(path):
    """Wczytuje punkty (sekundy od północy, natężenie) z pliku .xlsx/.xlsm/.csv (kolumna A: godzina, B: natężenie).
    Wiersze, których nie da się odczytać (np. nagłówek), są pomijane."""
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        try:
            for row in wb.active.iter_rows(values_only=True):
                if row and len(row) >= 2:
                    rows.append((row[0], row[1]))
        finally:
            wb.close()
    elif ext == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            for row in csv.reader(f, dialect):
                if len(row) >= 2:
                    rows.append((row[0], row[1]))
    else:
        raise ValueError("Obsługiwane formaty: .xlsx, .xlsm, .csv (starszy .xls zapisz jako .xlsx).")
    points = []
    for t_cell, v_cell in rows:
        t, v = parse_time_cell(t_cell), parse_number(v_cell)
        if t is not None and v is not None:
            points.append((t, v))
    if not points:
        raise ValueError("Nie znaleziono danych. Oczekiwane: kolumna A — godzina, kolumna B — natężenie [poj./h].")
    return points


class DemandProfile:
    """Natężenie ruchu [poj./h] w siatce 5-minutowej. Braki w środku są uzupełniane liniowo (jeśli włączono),
    a godziny przed pierwszym i po ostatnim punkcie są niedostępne."""
    STEP = 300  # s

    def __init__(self, points):
        slots = {}
        for t, v in points:
            slots.setdefault(int(round(t / self.STEP)), []).append(v)
        self.values = {k: float(np.mean(v)) for k, v in slots.items()}
        self.keys = sorted(self.values)
        self.first, self.last = self.keys[0], self.keys[-1]
        self.missing = [k for k in range(self.first, self.last + 1) if k not in self.values]
        self._xs = np.array(self.keys, dtype=float)
        self._ys = np.array([self.values[k] for k in self.keys], dtype=float)

    def rate(self, t):
        """Natężenie w chwili t (s od północy), interpolacja liniowa między znanymi punktami."""
        return float(np.interp(t / self.STEP, self._xs, self._ys))

    def missing_runs(self):
        """Ciągłe przedziały brakujących punktów jako [(od_s, do_s, liczba_punktów)]."""
        runs, start, prev = [], None, None
        for k in self.missing:
            if start is None:
                start = prev = k
            elif k == prev + 1:
                prev = k
            else:
                runs.append((start * self.STEP, prev * self.STEP, prev - start + 1))
                start = prev = k
        if start is not None:
            runs.append((start * self.STEP, prev * self.STEP, prev - start + 1))
        return runs

    def segments(self, interpolate):
        """Zakresy (od_s, do_s), w których wolno wybrać okno."""
        if interpolate:
            return [(self.first * self.STEP, self.last * self.STEP)]
        segs, start, prev = [], None, None
        for k in self.keys:
            if start is None:
                start = prev = k
            elif k == prev + 1:
                prev = k
            else:
                segs.append((start * self.STEP, prev * self.STEP))
                start = prev = k
        segs.append((start * self.STEP, prev * self.STEP))
        return segs

    def window_ok(self, t0, t1, interpolate):
        if t1 <= t0:
            return False, "Koniec okna musi być późniejszy niż początek."
        if t0 < self.first * self.STEP or t1 > self.last * self.STEP:
            return False, "Okno wykracza poza zakres danych (brak danych na początku lub końcu)."
        for a, b in self.segments(interpolate):
            if a <= t0 and t1 <= b:
                return True, ""
        return False, "Okno obejmuje brakujące punkty, a uzupełnianie braków jest wyłączone."


def load_od_pool(log):
    """Pula par start-cel (zapisywana w pliku i używana ponownie, dopóki sieć się nie zmieni)."""
    import subprocess
    fresh = (os.path.exists(OD_POOL_FILE)
             and os.path.getmtime(OD_POOL_FILE) >= os.path.getmtime(NET_FILE))
    if not fresh:
        log("Generuję pulę par start-cel (O-D)...")
        random_trips = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")
        subprocess.run([
            sys.executable, random_trips,
            "-n", NET_FILE, "-e", "500", "--period", "0.05",
            "--min-distance", "300", "--fringe-factor", "10", "--fringe-threshold", "10",
            "-o", OD_POOL_FILE,
        ], check=True)
    with open(OD_POOL_FILE) as f:
        content = f.read()
    pool = re.findall(r'<trip id="([^"]+)"[^>]*from="([^"]+)"[^>]*to="([^"]+)"[^>]*/>', content)
    if not pool:
        raise RuntimeError("Nie udało się wygenerować puli O-D.")
    return pool


def generate_window_routes(profile, t0, t1, warmup, multiplier, log):
    """Trasy dla okna [t0, t1] (s od północy): pojazdy pojawiają się według profilu natężenia (proces Poissona).
    Czas symulacji s = t - (t0 - warmup); w czasie rozgrzewki natężenie jest stałe i równe natężeniu z początku okna.
    Zwraca opis scenariusza dla run_simulation / replay_simulation."""
    import subprocess
    od_pool = load_od_pool(log)
    sim_end = warmup + (t1 - t0)
    offset = t0 - warmup
    hold_rate = profile.rate(t0) * multiplier

    rng = np.random.default_rng(42)   # stałe ziarno: wszyscy kandydaci widzą ten sam ruch
    trips = []
    s = 0.0
    while s < sim_end:
        width = min(60.0, sim_end - s)
        t_abs = offset + s + width / 2.0
        rate = hold_rate if t_abs < t0 else profile.rate(min(t_abs, t1)) * multiplier
        n = rng.poisson(rate * width / 3600.0)
        for depart in rng.uniform(s, s + width, size=n):
            _, from_edge, to_edge = od_pool[rng.integers(len(od_pool))]
            trips.append((float(depart), from_edge, to_edge))
        s += 60.0
    trips.sort()

    temp_trips = os.path.join(SCRIPT_DIR, "temp_window.trips.xml")
    with open(temp_trips, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        for i, (depart, from_edge, to_edge) in enumerate(trips):
            f.write(f'    <trip id="veh{i}" depart="{depart:.2f}" from="{from_edge}" to="{to_edge}"/>\n')
        f.write('</routes>\n')

    log(f"Okno {format_time_of_day(t0)}–{format_time_of_day(t1)}: {len(trips)} pojazdów "
        f"(średnio {len(trips) / (sim_end / 3600.0):.0f} poj./h), wyznaczam trasy...")
    subprocess.run([
        sumolib.checkBinary("duarouter"), "-n", NET_FILE, "--route-files", temp_trips,
        "-o", WINDOW_ROUTE_FILE, "--no-step-log", "--ignore-errors", "--remove-loops",
    ], check=True)

    return {
        "route_file": WINDOW_ROUTE_FILE,
        "end_time": float(sim_end),
        "measure_from": float(warmup),
        "clock_offset": float(offset),
        "preview_end": float(min(sim_end, warmup + 900)),
    }


# ============================================================
# --- Geometria sieci i widok rysowany w Qt ---
# ============================================================

class NetGeometry:
    """Sieć wczytana raz i współdzielona przez wszystkie widoki."""

    def __init__(self, net_file):
        self.net = sumolib.net.readNet(net_file)
        self.road_path = QPainterPath()      # pasy dostępne dla samochodów
        self.other_path = QPainterPath()     # pozostałe pasy (np. tory)
        for edge in self.net.getEdges():
            for lane in edge.getLanes():
                shape = [(p[0], p[1]) for p in lane.getShape()]
                if len(shape) < 2:
                    continue
                path = self.road_path if lane.allows("passenger") else self.other_path
                path.moveTo(shape[0][0], shape[0][1])
                for x, y in shape[1:]:
                    path.lineTo(x, y)
        self.junction_path = QPainterPath()
        for node in self.net.getNodes():
            shape = [(p[0], p[1]) for p in node.getShape()]
            if len(shape) >= 3:
                self.junction_path.addPolygon(QPolygonF([QPointF(x, y) for x, y in shape]))
                self.junction_path.closeSubpath()
        self.bounds = self.net.getBoundary()   # (xmin, ymin, xmax, ymax)
        self.car_tls_ids = static_car_relevant_tls(self.net)

    def tls_geometry(self, tls_ids):
        """Dla każdej sygnalizacji: pozycja etykiety i linie na końcach pasów dojazdowych."""
        result = {}
        for tid in tls_ids:
            try:
                tls = self.net.getTLSSecure(tid)
                conns = tls.getConnections()
            except Exception:
                continue
            xs, ys = [], []
            lane_ends = {}
            for in_lane, _out_lane, link_no in conns:
                if link_no < 0:
                    continue
                node_xy = in_lane.getEdge().getToNode().getCoord()
                xs.append(node_xy[0])
                ys.append(node_xy[1])
                lid = in_lane.getID()
                if lid in lane_ends:
                    lane_ends[lid]["links"].append(link_no)
                    continue
                shape = in_lane.getShape()
                if len(shape) < 2:
                    continue
                x0, y0 = shape[-2][0], shape[-2][1]
                x1, y1 = shape[-1][0], shape[-1][1]
                length = math.hypot(x1 - x0, y1 - y0)
                if length == 0:
                    continue
                dx, dy = (x1 - x0) / length, (y1 - y0) / length
                ex, ey = x1 - dx * 0.5, y1 - dy * 0.5
                hw = in_lane.getWidth() / 2.0
                nx, ny = -dy, dx
                lane_ends[lid] = {
                    "a": (ex + nx * hw, ey + ny * hw),
                    "b": (ex - nx * hw, ey - ny * hw),
                    "links": [link_no],
                }
            label = (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else None
            result[tid] = {"label": label, "ends": list(lane_ends.values())}
        return result


class NetworkView(QWidget):
    """Widok sieci: drogi, numery skrzyżowań, światła i pojazdy. Kółko myszy = zoom, przeciąganie = przesuwanie,
    dwuklik = dopasuj do okna."""

    viewChanged = Signal(float, float, float)   # scale, cx, cy (do synchronizacji dwóch widoków)

    COLOR_BG = QColor("#1b1d21")
    COLOR_ROAD = QColor("#5d636b")
    COLOR_OTHER = QColor("#3a3e44")
    COLOR_JUNCTION = QColor("#4a4f56")
    COLOR_LABEL = QColor("#f2f2f2")
    COLOR_MOVING = QColor("#3ddc84")
    COLOR_STOPPED = QColor("#ff5252")
    SIGNAL_COLORS = {3: QColor("#00e676"), 2: QColor("#ffd600"), 1: QColor("#ff1744"), 0: QColor("#888888")}

    def __init__(self, geom, title=""):
        super().__init__()
        self.geom = geom
        self.title = title
        self.tls_geo = {}
        self.frame = None
        self.setMinimumSize(300, 240)

        xmin, ymin, xmax, ymax = geom.bounds
        self.cx = (xmin + xmax) / 2.0
        self.cy = (ymin + ymax) / 2.0
        self.scale = 1.0
        self._auto_fit = True
        self._drag = None
        self._static = None
        self._static_dirty = True

    # ---------- dane ----------

    def set_tls(self, tls_ids):
        self.tls_geo = self.geom.tls_geometry(tls_ids)
        self._static_dirty = True
        self.update()

    def set_frame(self, frame):
        self.frame = frame
        self.update()

    def clear_frame(self):
        self.frame = None
        self.update()

    # ---------- widok ----------

    def _fit(self):
        xmin, ymin, xmax, ymax = self.geom.bounds
        w, h = max(self.width(), 1), max(self.height(), 1)
        self.scale = min(w / max(xmax - xmin, 1.0), h / max(ymax - ymin, 1.0)) * 0.95
        self.cx = (xmin + xmax) / 2.0
        self.cy = (ymin + ymax) / 2.0
        self._static_dirty = True

    def set_view(self, scale, cx, cy):
        self.scale, self.cx, self.cy = scale, cx, cy
        self._auto_fit = False
        self._static_dirty = True
        self.update()

    def _changed(self):
        self._static_dirty = True
        self.update()
        self.viewChanged.emit(self.scale, self.cx, self.cy)

    def resizeEvent(self, event):
        if self._auto_fit:
            self._fit()
        self._static_dirty = True
        super().resizeEvent(event)

    def wheelEvent(self, event):
        pos = event.position()
        wx = self.cx + (pos.x() - self.width() / 2.0) / self.scale
        wy = self.cy - (pos.y() - self.height() / 2.0) / self.scale
        factor = 1.15 ** (event.angleDelta().y() / 120.0)
        self.scale = min(max(self.scale * factor, 0.005), 50.0)
        self.cx = wx - (pos.x() - self.width() / 2.0) / self.scale
        self.cy = wy + (pos.y() - self.height() / 2.0) / self.scale
        self._auto_fit = False
        self._changed()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            pos = event.position()
            self.cx -= (pos.x() - self._drag.x()) / self.scale
            self.cy += (pos.y() - self._drag.y()) / self.scale
            self._drag = pos
            self._auto_fit = False
            self._changed()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        self._auto_fit = True
        self._fit()
        self._changed()

    # ---------- rysowanie ----------

    def _to_screen(self, x, y):
        return ((x - self.cx) * self.scale + self.width() / 2.0,
                self.height() / 2.0 - (y - self.cy) * self.scale)

    def _world_transform(self):
        t = QTransform()
        t.translate(self.width() / 2.0, self.height() / 2.0)
        t.scale(self.scale, -self.scale)
        t.translate(-self.cx, -self.cy)
        return t

    @staticmethod
    def _short(tid):
        return tid if len(tid) <= 16 else tid[:15] + "…"

    def _render_static(self):
        dpr = self.devicePixelRatioF()
        w, h = max(self.width(), 1), max(self.height(), 1)
        pm = QPixmap(int(w * dpr), int(h * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(self.COLOR_BG)

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setTransform(self._world_transform())
        p.fillPath(self.geom.junction_path, QBrush(self.COLOR_JUNCTION))
        pen = QPen(self.COLOR_OTHER, max(1.0, 3.2 * self.scale))
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.FlatCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(self.geom.other_path)
        pen.setColor(self.COLOR_ROAD)
        p.setPen(pen)
        p.drawPath(self.geom.road_path)
        p.resetTransform()

        # numery skrzyżowań
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()
        for tid, g in self.tls_geo.items():
            if g["label"] is None:
                continue
            sx, sy = self._to_screen(*g["label"])
            text = self._short(tid)
            x = sx - fm.horizontalAdvance(text) / 2.0
            y = sy - 14
            p.setPen(QColor(0, 0, 0, 230))
            for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                p.drawText(QPointF(x + ox, y + oy), text)
            p.setPen(self.COLOR_LABEL)
            p.drawText(QPointF(x, y), text)
        p.end()
        self._static = pm

    @classmethod
    def _signal_color(cls, state, links):
        best = 0
        for k in links:
            if k >= len(state):
                continue
            c = state[k]
            if c in "Gg":
                rank = 3
            elif c in "Yy":
                rank = 2
            elif c in "Rr":
                rank = 1
            else:
                rank = 0
            best = max(best, rank)
        return cls.SIGNAL_COLORS[best]

    def paintEvent(self, event):
        if self._static is None or self._static_dirty:
            self._render_static()
            self._static_dirty = False
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.drawPixmap(0, 0, self._static)

        frame = self.frame
        n_veh = n_stopped = 0
        if frame is not None:
            # światła: kropka na końcu każdego pasa dojazdowego (kolor = najbardziej "otwarty" kierunek z pasa)
            r_sig = max(3.2, 1.3 * self.scale)
            p.setPen(QPen(QColor(0, 0, 0, 200), 1))
            for tid, g in self.tls_geo.items():
                state = frame["tls"].get(tid)
                if not state:
                    continue
                for end in g["ends"]:
                    p.setBrush(self._signal_color(state, end["links"]))
                    mx = (end["a"][0] + end["b"][0]) / 2.0
                    my = (end["a"][1] + end["b"][1]) / 2.0
                    x, y = self._to_screen(mx, my)
                    p.drawEllipse(QPointF(x, y), r_sig, r_sig)

            # pojazdy
            veh = frame["veh"]
            n_veh = len(veh)
            if n_veh:
                sx = (veh[:, 0] - self.cx) * self.scale + self.width() / 2.0
                sy = self.height() / 2.0 - (veh[:, 1] - self.cy) * self.scale
                moving = veh[:, 2] > 0.1
                n_stopped = int((~moving).sum())
                r = min(6.0, max(1.8, 1.6 * self.scale))
                p.setPen(Qt.NoPen)
                for mask, color in ((moving, self.COLOR_MOVING), (~moving, self.COLOR_STOPPED)):
                    p.setBrush(color)
                    for x, y in zip(sx[mask], sy[mask]):
                        p.drawEllipse(QPointF(x, y), r, r)

        # pasek informacyjny
        p.fillRect(0, 0, self.width(), 22, QColor(0, 0, 0, 150))
        p.setPen(QColor("#e8e8e8"))
        font = QFont()
        font.setPointSize(9)
        p.setFont(font)
        info = self.title
        if frame is not None:
            info += f"   {format_time_of_day(frame['t'])}   pojazdów: {n_veh}   stoi: {n_stopped}"
        p.drawText(8, 16, info)
        p.end()


# ============================================================
# --- Worker trybu zwykłego ---
# ============================================================

def _batch_eval_task(args):
    """Wykonywana w osobnym procesie: pełna symulacja kandydata z danym ziarnem.
    Zwraca (suma czasu oczekiwania, śr. prędkość w m/s)."""
    vector, layout, route_file, end_time, measure_from, seed = args
    return run_simulation(vector, layout, route_file=route_file, end_time=end_time,
                          measure_from=measure_from, control=_WORKER_CONTROL, seed=seed)


class BatchOptimizationWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int, float)
    frame_signal = Signal(object)
    finished_signal = Signal(object, float, float, float)

    def __init__(self, algo_name, n_particles, n_iterations, bound_lo, bound_hi,
                 show_best_each_iter, speed_factor, tls_candidates, window_spec=None,
                 noise=8.0, patience=8, n_processes=2):
        super().__init__()
        self.algo_name = algo_name
        self.cfg = {"n": n_particles, "iterations": n_iterations, "noise": noise,
                    "n_random": 1, "patience": patience}
        self.bounds = (bound_lo, bound_hi)
        self.show_best_each_iter = show_best_each_iter
        self.speed_factor = speed_factor
        self.tls_candidates = tls_candidates
        self.window_spec = window_spec    # wymagane: profil natężenia, okno godzin, rozgrzewka, mnożnik
        self.n_processes = n_processes
        # zdarzenia współdzielone z procesami potomnymi: pauza i anulowanie działają też w trakcie symulacji
        self._ctx = mp.get_context("spawn")
        running = self._ctx.Event()
        running.set()
        self.control = RunControl(running, self._ctx.Event())
        self.best_so_far = None
        self._pool = None
        self._scenario = None
        self._layout = None

    def log(self, msg):
        self.log_signal.emit(str(msg))

    def run(self):
        result = None
        try:
            result = self._run()
        except StopRequested:
            if self.best_so_far is not None:
                self.log(f"Anulowano. Najlepszy wynik do tej chwili: {self.best_so_far:.0f} (nie zapisano).")
            else:
                self.log("Anulowano — nic nie zapisano.")
        except Exception as e:
            self.log(f"BŁĄD: {type(e).__name__}: {e}")
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
        if result is None:
            self.finished_signal.emit(None, 0.0, 0.0, 0.0)
        else:
            self.finished_signal.emit(*result)

    def _evaluate(self, vectors, seeds):
        """Pełne symulacje (wektor x ziarno) rozdzielane między procesy.
        Zwraca dla każdego wektora (średnia suma czasu oczekiwania, średnia prędkość) po ziarnach.
        Wektor None oznacza program z sieci."""
        self.control.checkpoint()
        sc, layout = self._scenario, self._layout
        tasks = [(v, layout, sc["route_file"], sc["end_time"], sc["measure_from"], seed)
                 for v in vectors for seed in seeds]
        results = list(self._pool.map(_batch_eval_task, tasks))
        k = len(seeds)
        return [(float(np.mean([r[0] for r in results[i * k:(i + 1) * k]])),
                 float(np.mean([r[1] for r in results[i * k:(i + 1) * k]])))
                for i in range(len(vectors))]

    def _run(self):
        ws = self.window_spec
        self.log(f"Okno godzin: {format_time_of_day(ws['t0'])}–{format_time_of_day(ws['t1'])}, "
                 f"rozgrzewka {ws['warmup'] / 60:.0f} min, mnożnik natężenia {ws['multiplier']:.2f}")
        scenario = generate_window_routes(ws["profile"], ws["t0"], ws["t1"],
                                          ws["warmup"], ws["multiplier"], self.log)
        meta = {"window": f"{format_time_of_day(ws['t0'])}-{format_time_of_day(ws['t1'])}",
                "warmup_s": ws["warmup"], "multiplier": ws["multiplier"]}
        self._scenario = scenario
        self.control.checkpoint()

        conn = start_sumo(["sumo", "-n", NET_FILE, "-r", scenario["route_file"], "--no-step-log"], "setup")
        try:
            known = set(conn.trafficlight.getIDList())
            car_relevant_tls = [t for t in self.tls_candidates if t in known]
            if not car_relevant_tls:
                self.log("Nie znaleziono sygnalizacji sterujących ruchem samochodowym.")
                return None
            self.log(f"Skrzyżowania: {car_relevant_tls}")
            layout, dims = build_tls_layout(car_relevant_tls, self.log, conn, min_tunable=self.bounds[0])
            baseline_vector = build_baseline_vector(layout, car_relevant_tls, conn)
        finally:
            close_conn(conn)

        if dims == 0:
            self.log("Brak faz do optymalizacji (wszystkie sygnalizacje są akomodacyjne albo nie mają faz zielonych).")
            return None
        self._layout = layout
        self.cfg["groups"] = [(st, st + len(ix)) for st, ix in layout.values() if ix]
        # granice per faza: dolna = minimum z okna, górna = większa z (maksimum z okna, wartość z sieci),
        # dzięki temu ustawienia bazowe nigdy nie są obcinane
        lo_vec = np.full(dims, float(self.bounds[0]))
        hi_vec = np.maximum(float(self.bounds[1]), baseline_vector)

        def replay(vector, end_time):
            replay_simulation(vector, layout, car_relevant_tls, self.frame_signal.emit, self.speed_factor,
                              end_time=end_time, route_file=scenario["route_file"],
                              clock_offset=scenario["clock_offset"], control=self.control,
                              seed=EVAL_SEEDS[0])

        self.log(f"Wymiary przeszukiwania: {dims} (skrzyżowań do strojenia: {len(self.cfg['groups'])})")
        self._pool = make_pool(self._ctx, self.n_processes, self.control)
        self.log(f"Procesy równoległe: {self.n_processes}; ziarna oceny: {EVAL_SEEDS}, ziarna walidacji: {VALIDATION_SEEDS}")

        # program z sieci oraz te same ustawienia zastosowane przez TraCI (kontrola spójności), po 2 ziarnach
        self.log("Liczę wynik bazowy...")
        (baseline_wait, baseline_speed), (control_wait, _) = self._evaluate([None, baseline_vector], EVAL_SEEDS)
        self.log(f"Bazowy całkowity czas oczekiwania (średnia z {len(EVAL_SEEDS)} ziaren): {baseline_wait:.3f}")
        start_score = baseline_wait
        if baseline_wait and abs(control_wait / baseline_wait - 1) > 0.01:
            self.log(f"UWAGA: ustawienia bazowe zastosowane przez TraCI dają inny wynik ({control_wait:.0f}, "
                     f"{(control_wait / baseline_wait - 1) * 100:+.1f}% względem programu z sieci). "
                     f"Kandydaci są porównywani z wynikiem kontrolnym.")
            start_score = control_wait
        else:
            self.log(f"Kontrola spójności: ustawienia bazowe zastosowane przez TraCI dają ten sam wynik ({control_wait:.0f}).")
        self.best_so_far = start_score
        ref = start_score if start_score else 1.0

        iter_scores = []
        state = {"iters": 0, "warned": False}

        def evaluate_batch(vectors):
            out = [r[0] for r in self._evaluate(vectors, EVAL_SEEDS)]
            iter_scores.extend(out)
            return out

        last_shown_score = [start_score]

        def on_iteration(it, total, score, best_vector):
            state["iters"] = it
            self.best_so_far = score
            self.progress_signal.emit(it, total, score)
            line = (f"[{self.algo_name}] Iteracja {it}/{total} — najlepszy wynik: {score:.0f} "
                    f"({(score / ref - 1) * 100:+.1f}% względem bazowego)")
            if iter_scores:
                sc = np.array(iter_scores)
                line += (f" | oceny w iteracji: min {sc.min():.0f}, śr. {sc.mean():.0f}, maks {sc.max():.0f}"
                         f" | lepszych od bazowego: {int((sc < start_score).sum())}/{len(sc)}")
                if len(sc) >= 3 and np.ptp(sc) == 0 and not state["warned"]:
                    state["warned"] = True
                    self.log("UWAGA: wszystkie oceny w tej iteracji są identyczne — zmiana czasów faz nie wpływa "
                             "na wynik. Sprawdź, czy sygnalizacje są stałoczasowe i czy ruch przejeżdża przez te skrzyżowania.")
                iter_scores.clear()
            self.log(line)
            if self.show_best_each_iter and score < last_shown_score[0]:
                last_shown_score[0] = score
                self.log("  -> odtwarzam aktualnie najlepszy wynik w podglądzie...")
                replay(best_vector, scenario["preview_end"])

        best_vector, best_score = ALGORITHMS[self.algo_name](
            evaluate_batch, dims, (lo_vec, hi_vec), baseline_vector, start_score, self.cfg, on_iteration)

        if state["iters"] < self.cfg["iterations"]:
            self.log(f"Zatrzymano wcześniej po {state['iters']} iteracjach: najlepszy wynik nie poprawiał się "
                     f"(parametr „Cierpliwość”: {self.cfg['patience']}).")

        # warunek przyjęcia: lepszy od ustawień bieżących o co najmniej MIN_GAIN i potwierdzony na nowych ziarnach
        final_vector, final_score = baseline_vector, start_score
        if best_score >= start_score * (1 - MIN_GAIN):
            self.log(f"Najlepszy wynik ({best_score:.0f}) nie jest lepszy od bazowego ({start_score:.0f}) "
                     f"o co najmniej {MIN_GAIN * 100:.0f}%, więc zostają ustawienia bazowe. "
                     "Spróbuj większej liczby osobników/iteracji, mniejszego rozrzutu początkowego lub innego algorytmu.")
        else:
            self.log(f"Walidacja na nowych ziarnach {VALIDATION_SEEDS}...")
            (ctrl_v, _), (best_v, _) = self._evaluate([baseline_vector, best_vector], VALIDATION_SEEDS)
            if best_v < ctrl_v * (1 - MIN_GAIN):
                self.log(f"Zmiana potwierdzona na nowych ziarnach ({best_v:.0f} vs {ctrl_v:.0f}) — przyjmuję.")
                final_vector, final_score = best_vector, best_score
            else:
                self.log(f"Zmiana odrzucona po walidacji ({best_v:.0f} vs {ctrl_v:.0f}) — zostają ustawienia bazowe.")

        final_vector = np.round(final_vector, 1)
        self.log(f"Wybrane ustawienia ({self.algo_name}): {final_score:.3f}")
        write_tls_settings(layout, final_vector, OUTPUT_DIR, self.log, scenario["route_file"], meta=meta)

        (opt_wait, opt_speed), = self._evaluate([final_vector], EVAL_SEEDS)
        self.log(f"Zoptymalizowany czas oczekiwania (średnia z {len(EVAL_SEEDS)} ziaren): {opt_wait:.3f}, "
                 f"prędkość: {opt_speed * 3.6:.3f} km/h (bazowa: {baseline_speed * 3.6:.3f} km/h)")
        self.log("Odtwarzam finalną symulację w podglądzie...")
        replay(final_vector, scenario["end_time"])
        return final_vector, final_score, opt_wait, opt_speed


# ============================================================
# --- Tryb ciągły: statystyki, zadanie procesu, worker ---
# ============================================================

class BinStats:
    """Statystyki ruchu w jednym przedziale 15-minutowym (z subskrypcji TraCI)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stopped_steps = 0
        self.moving_steps = 0
        self.speed_sum = 0.0
        self.vehicles = set()

    def add_from(self, results):
        for vid, data in results.items():
            speed = data.get(tc.VAR_SPEED)
            if speed is None:
                continue
            self.vehicles.add(vid)
            if speed > 0.1:
                self.moving_steps += 1
                self.speed_sum += speed
            else:
                self.stopped_steps += 1

    def summary(self):
        # krok = 1 s, więc liczba kroków = liczba sekund
        n = len(self.vehicles)
        avg_stop = self.stopped_steps / n if n else 0.0
        avg_move = self.moving_steps / n if n else 0.0
        avg_speed_kmh = (self.speed_sum / self.moving_steps * 3.6) if self.moving_steps else 0.0
        return (avg_stop, avg_move, avg_speed_kmh)


def _eval_task(args):
    """Wykonywana w osobnym procesie: jedna krótka symulacja od zapisanego stanu.
    Zwraca liczbę sekund stania pojazdów w oknie oceny."""
    candidate, layout, route_file, state_path, seed, window = args
    cmd = ["sumo", "-n", NET_FILE, "-r", route_file, "--no-step-log",
           "--seed", str(seed), "--load-state", state_path]
    conn = start_sumo(cmd, "eval")
    try:
        apply_durations(layout, candidate, conn)
        stopped = 0
        for _ in range(window):
            if _WORKER_CONTROL is not None:
                _WORKER_CONTROL.checkpoint()
            conn.simulationStep()
            for veh_id in conn.vehicle.getIDList():
                if conn.vehicle.getSpeed(veh_id) <= 0.1:
                    stopped += 1
        return float(stopped)
    finally:
        close_conn(conn)


class ContinuousWorker(QThread):
    log_signal = Signal(str)
    bin_signal = Signal(int, str, object, object)
    frame_signal = Signal(object, object)      # (klatka zoptymalizowanej, klatka bazowej)
    finished_signal = Signal()

    def __init__(self, algo_name, n_individuals, n_iterations, n_processes,
                 opt_interval, eval_window, noise, bound_lo, bound_hi,
                 speed_factor, tls_candidates, window_spec):
        super().__init__()
        self.algo_name = algo_name
        self.cfg = {"n": n_individuals, "iterations": n_iterations, "noise": noise, "n_random": 1}
        self.n_processes = n_processes
        self.opt_interval = opt_interval
        self.eval_window = eval_window
        self.bounds = (bound_lo, bound_hi)
        self.speed_factor = speed_factor
        self.tls_candidates = tls_candidates
        self.window_spec = window_spec
        self._route_file = None
        self._ctx = mp.get_context("spawn")
        running = self._ctx.Event()
        running.set()
        self.control = RunControl(running, self._ctx.Event())
        self._pool = None

    def log(self, msg):
        self.log_signal.emit(str(msg))

    def request_stop(self):
        self.control.cancel()

    def _evaluate_batch(self, vectors, layout, state_path, seeds):
        """Zadania (wektor x ziarno) rozdzielane między procesy; wynik wektora = średnia po ziarnach."""
        self.control.checkpoint()
        tasks = [(vec, layout, self._route_file, state_path, seed, self.eval_window) for vec in vectors for seed in seeds]
        results = list(self._pool.map(_eval_task, tasks))
        k = len(seeds)
        return [float(np.mean(results[i * k:(i + 1) * k])) for i in range(len(vectors))]

    def _tune(self, state_path, layout, dims, current, lo_vec, hi_vec):
        """Jedno dostrojenie: zwraca nowy wektor albo ten sam obiekt `current`, jeśli bez zmian."""
        control = self._evaluate_batch([current], layout, state_path, EVAL_SEEDS)[0]
        if control < MIN_SCORE_TO_TUNE:
            self.log(f"     mały ruch (stanie: {control:.0f} s w oknie) — pomijam dostrajanie")
            return current
        self.log(f"     wynik obecnych ustawień: {control:.0f}")

        def evaluate_batch(vectors):
            return self._evaluate_batch(vectors, layout, state_path, EVAL_SEEDS)

        def on_iteration(it, total, score, _vec):
            self.log(f"     [{self.algo_name}] iteracja {it}/{total}: najlepszy wynik {score:.0f}")

        best_vec, best_score = ALGORITHMS[self.algo_name](
            evaluate_batch, dims, (lo_vec, hi_vec), current, control, self.cfg, on_iteration)

        if best_score >= control * (1 - MIN_GAIN):
            self.log("     bez zmian (rój nie znalazł wyraźnie lepszych ustawień)")
            return current

        # potwierdzenie na świeżych ziarnach: odrzuca dopasowanie do szumu
        ctrl_v, best_v = self._evaluate_batch([current, best_vec], layout, state_path, VALIDATION_SEEDS)
        if best_v < ctrl_v * (1 - MIN_GAIN):
            self.log(f"     zmiana potwierdzona na nowych ziarnach ({best_v:.0f} vs {ctrl_v:.0f}) — stosuję")
            return best_vec
        self.log(f"     zmiana odrzucona po walidacji ({best_v:.0f} vs {ctrl_v:.0f})")
        return current

    def run(self):
        live = None
        baseline = None
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            ws = self.window_spec
            self.log(f"Okno godzin: {format_time_of_day(ws['t0'])}–{format_time_of_day(ws['t1'])}, "
                     f"rozgrzewka {ws['warmup'] / 60:.0f} min, mnożnik natężenia {ws['multiplier']:.2f}")
            scenario = generate_window_routes(ws["profile"], ws["t0"], ws["t1"],
                                              ws["warmup"], ws["multiplier"], self.log)
            route_file = scenario["route_file"]
            self._route_file = route_file
            sim_end, warm, clock_offset = scenario["end_time"], scenario["measure_from"], scenario["clock_offset"]
            t0 = ws["t0"]
            self.control.checkpoint()

            lo, hi = self.bounds

            live = start_sumo(["sumo", "-n", NET_FILE, "-r", route_file,
                               "--no-step-log", "--seed", SIM_SEED], "live")
            baseline = start_sumo(["sumo", "-n", NET_FILE, "-r", route_file,
                                   "--no-step-log", "--seed", SIM_SEED], "baseline")

            known = set(live.trafficlight.getIDList())
            tls_ids = [t for t in self.tls_candidates if t in known]
            if not tls_ids:
                self.log("Nie znaleziono sygnalizacji sterujących ruchem samochodowym.")
                return

            self.log(f"Skrzyżowania: {tls_ids}")
            layout, dims = build_tls_layout(tls_ids, self.log, live, min_tunable=lo)
            if dims == 0:
                self.log("Brak faz do strojenia przy tej minimalnej długości fazy.")
                return
            self.cfg["groups"] = [(st, st + len(ix)) for st, ix in layout.values() if ix]

            original = build_baseline_vector(layout, tls_ids, live)
            current_durations = original.copy()
            # granice per faza: wokół oryginału, w ramach globalnych granic
            lo_vec = np.minimum(original, np.maximum(lo, original - MAX_DEVIATION))
            hi_vec = np.maximum(original, np.minimum(hi, original + MAX_DEVIATION))

            self._pool = make_pool(self._ctx, self.n_processes, self.control)
            self.log(f"Algorytm: {self.algo_name}, procesy równoległe: {self.n_processes}")

            live_tracker = VehicleTracker(live)
            base_tracker = VehicleTracker(baseline)
            pacer = FramePacer(self.speed_factor)
            opt_stats = BinStats()
            base_stats = BinStats()
            current_bin = 0
            next_opt_time = warm + self.opt_interval   # dostrajanie zaczyna się po rozgrzewce

            self.log(f"Rozpoczynam ciągłą symulację {format_time_of_day(t0)} - {format_time_of_day(ws['t1'])} "
                     f"(wersja zoptymalizowana + bazowa)...")

            while True:
                if self.control.checkpoint():
                    pacer.reset(live.simulation.getTime())
                live.simulationStep()
                baseline.simulationStep()
                t = live.simulation.getTime()

                live_res = live_tracker.update()
                base_res = base_tracker.update()
                if t > warm:                             # rozgrzewka nie wchodzi do statystyk
                    opt_stats.add_from(live_res)
                    base_stats.add_from(base_res)

                if t - warm >= (current_bin + 1) * BIN_SECONDS:
                    label = format_time_of_day(t0 + current_bin * BIN_SECONDS)
                    opt = opt_stats.summary()
                    base = base_stats.summary()
                    self.bin_signal.emit(current_bin, label, opt, base)
                    self.log(f"[{label}] zoptymalizowana: postój {opt[0]:.1f}s, ruch {opt[1]:.1f}s, {opt[2]:.1f} km/h | "
                             f"bazowa: postój {base[0]:.1f}s, ruch {base[1]:.1f}s, {base[2]:.1f} km/h")
                    opt_stats.reset()
                    base_stats.reset()
                    current_bin += 1

                if pacer.due():
                    self.frame_signal.emit(build_frame(live, live_res, tls_ids, t + clock_offset),
                                           build_frame(baseline, base_res, tls_ids, t + clock_offset))
                pacer.throttle(t)

                if next_opt_time <= t < sim_end - 1:
                    next_opt_time += self.opt_interval
                    self.log(f"  -> dostrajam światła ({self.algo_name}) o {format_time_of_day(t + clock_offset)}...")

                    state_path = os.path.join(STATE_DIR, f"state_{int(t)}.xml")
                    live.simulation.saveState(state_path)

                    try:
                        new_durations = self._tune(state_path, layout, dims, current_durations, lo_vec, hi_vec)
                    except StopRequested:
                        raise
                    except Exception as e:
                        self.log(f"     dostrajanie nie powiodło się: {type(e).__name__}: {e}")
                        new_durations = current_durations

                    if new_durations is not current_durations:
                        current_durations = new_durations
                        apply_durations(layout, current_durations, live)

                    try:
                        os.remove(state_path)
                    except OSError:
                        pass
                    pacer.reset(t)   # po dostrajaniu nie nadrabiamy zaległości w czasie rzeczywistym

                if t >= sim_end:
                    self.log("Osiągnięto koniec okna godzin.")
                    break

            if opt_stats.vehicles or base_stats.vehicles:      # niepełny ostatni przedział
                self.bin_signal.emit(current_bin, format_time_of_day(t0 + current_bin * BIN_SECONDS),
                                     opt_stats.summary(), base_stats.summary())

        except StopRequested:
            self.log("Zatrzymano na żądanie.")
        except traci.exceptions.FatalTraCIError as e:
            self.log(f"Połączenie z SUMO zostało przerwane: {e}")
        except Exception as e:
            self.log(f"BŁĄD: {type(e).__name__}: {e}")
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
            for conn in (live, baseline):
                if conn is not None:
                    close_conn(conn)
            self.finished_signal.emit()


def make_time_table(name_a, name_b):
    """Tabela średnich wartości w przedziałach 15-minutowych (wiersze dodaje reset_time_table)."""
    table = QTableWidget()
    table.setColumnCount(7)
    table.setHorizontalHeaderLabels([
        "Godzina",
        f"{name_a}: śr. postój [s/poj.]", f"{name_a}: śr. ruch [s/poj.]", f"{name_a}: śr. prędkość [km/h]",
        f"{name_b}: śr. postój [s/poj.]", f"{name_b}: śr. ruch [s/poj.]", f"{name_b}: śr. prędkość [km/h]",
    ])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return table


def reset_time_table(table, n_bins, t0):
    """Przygotowuje n_bins wierszy z godzinami zaczynającymi się od t0 (s od północy)."""
    table.setRowCount(n_bins)
    for i in range(n_bins):
        table.setItem(i, 0, QTableWidgetItem(format_time_of_day(t0 + i * BIN_SECONDS)))
        for col in range(1, 7):
            table.setItem(i, col, QTableWidgetItem(""))


# ============================================================
# --- Sygnalizacja akomodacyjna: worker ---
# ============================================================

class ActuatedWorker(QThread):
    """Symulacja z sygnalizacją akomodacyjną (detektory, przedłużanie zielonego) obok wersji stałoczasowej."""
    log_signal = Signal(str)
    bin_signal = Signal(int, str, object, object)
    frame_signal = Signal(object, object)      # (klatka akomodacyjnej, klatka stałoczasowej)
    summary_signal = Signal(object, object)    # statystyki z całego przebiegu
    finished_signal = Signal()

    def __init__(self, window_spec, params, speed_factor, tls_candidates):
        super().__init__()
        self.window_spec = window_spec
        self.params = params
        self.speed_factor = speed_factor
        self.tls_candidates = tls_candidates
        self.control = RunControl()

    def log(self, msg):
        self.log_signal.emit(str(msg))

    def request_stop(self):
        self.control.cancel()

    def run(self):
        act = None
        base = None
        try:
            ws = self.window_spec
            self.log(f"Okno godzin: {format_time_of_day(ws['t0'])}–{format_time_of_day(ws['t1'])}, "
                     f"rozgrzewka {ws['warmup'] / 60:.0f} min, mnożnik natężenia {ws['multiplier']:.2f}")
            scenario = generate_window_routes(ws["profile"], ws["t0"], ws["t1"],
                                              ws["warmup"], ws["multiplier"], self.log)
            route_file = scenario["route_file"]
            sim_end, warm, clock_offset = scenario["end_time"], scenario["measure_from"], scenario["clock_offset"]
            t0 = ws["t0"]
            self.control.checkpoint()

            # 1) odczyt istniejących programów i zapis pliku z programem akomodacyjnym
            setup = start_sumo(["sumo", "-n", NET_FILE, "-r", route_file, "--no-step-log"], "act_setup")
            try:
                known = set(setup.trafficlight.getIDList())
                tls_ids = [t for t in self.tls_candidates if t in known]
                if not tls_ids:
                    self.log("Nie znaleziono sygnalizacji sterujących ruchem samochodowym.")
                    return
                self.log(f"Skrzyżowania: {tls_ids}")
                self.log("Program akomodacyjny (fazy zielone dostają zakres min-max):")
                write_actuated_additional(tls_ids, setup, ACTUATED_ADD_FILE, self.params, self.log)
            finally:
                close_conn(setup)

            # 2) dwie symulacje z tym samym ziarnem: akomodacyjna i stałoczasowa
            act = start_sumo(["sumo", "-n", NET_FILE, "-r", route_file, "-a", ACTUATED_ADD_FILE,
                              "--no-step-log", "--seed", SIM_SEED], "act")
            base = start_sumo(["sumo", "-n", NET_FILE, "-r", route_file,
                               "--no-step-log", "--seed", SIM_SEED], "act_base")

            wrong = []
            for tid in tls_ids:
                act.trafficlight.setProgram(tid, ACT_PROGRAM_ID)
                if act.trafficlight.getProgram(tid) != ACT_PROGRAM_ID:
                    wrong.append(tid)
            if wrong:
                self.log(f"UWAGA: nie udało się włączyć programu akomodacyjnego dla: {wrong}")
            else:
                self.log(f"Program akomodacyjny aktywny na {len(tls_ids)} sygnalizacjach.")

            act_tracker = VehicleTracker(act)
            base_tracker = VehicleTracker(base)
            pacer = FramePacer(self.speed_factor)
            act_bin, base_bin = BinStats(), BinStats()
            act_total, base_total = BinStats(), BinStats()
            current_bin = 0

            self.log("Rozpoczynam symulację (akomodacyjna + stałoczasowa)...")

            while True:
                try:
                    if self.control.checkpoint():
                        pacer.reset(act.simulation.getTime())
                except StopRequested:
                    self.log("Zatrzymano na żądanie.")
                    break
                act.simulationStep()
                base.simulationStep()
                t = act.simulation.getTime()

                a_res = act_tracker.update()
                b_res = base_tracker.update()
                if t > warm:                             # rozgrzewka nie wchodzi do statystyk
                    act_bin.add_from(a_res)
                    act_total.add_from(a_res)
                    base_bin.add_from(b_res)
                    base_total.add_from(b_res)

                if t - warm >= (current_bin + 1) * BIN_SECONDS:
                    label = format_time_of_day(t0 + current_bin * BIN_SECONDS)
                    a, b = act_bin.summary(), base_bin.summary()
                    self.bin_signal.emit(current_bin, label, a, b)
                    self.log(f"[{label}] akomodacyjna: postój {a[0]:.1f}s, ruch {a[1]:.1f}s, {a[2]:.1f} km/h | "
                             f"stałoczasowa: postój {b[0]:.1f}s, ruch {b[1]:.1f}s, {b[2]:.1f} km/h")
                    act_bin.reset()
                    base_bin.reset()
                    current_bin += 1

                if pacer.due():
                    self.frame_signal.emit(build_frame(act, a_res, tls_ids, t + clock_offset),
                                           build_frame(base, b_res, tls_ids, t + clock_offset))
                pacer.throttle(t)

                if t >= sim_end:
                    self.log("Osiągnięto koniec okna godzin.")
                    break

            # niepełny ostatni przedział
            if act_bin.vehicles or base_bin.vehicles:
                self.bin_signal.emit(current_bin, format_time_of_day(t0 + current_bin * BIN_SECONDS),
                                     act_bin.summary(), base_bin.summary())
            self.summary_signal.emit(act_total.summary(), base_total.summary())

        except StopRequested:
            self.log("Zatrzymano na żądanie.")
        except traci.exceptions.FatalTraCIError as e:
            self.log(f"Połączenie z SUMO zostało przerwane: {e}")
        except Exception as e:
            self.log(f"BŁĄD: {type(e).__name__}: {e}")
        finally:
            for conn in (act, base):
                if conn is not None:
                    close_conn(conn)
            self.finished_signal.emit()


# ============================================================
# --- Panel danych o natężeniu ruchu (wymagany we wszystkich trybach) ---
# ============================================================

class DemandPanel(QGroupBox):
    """Wczytywanie natężenia z Excela/CSV, wybór okna godzin, mnożnik, rozgrzewka i wykres.
    Dopóki dane nie są wczytane, a okno poprawne, spec() zwraca None i tryb nie może ruszyć."""
    changed = Signal()
    profile_loaded = Signal(object, str)      # profil (albo None) i opis pliku — do synchronizacji paneli

    def __init__(self, title="Natężenie ruchu i okno godzin (wymagane)"):
        super().__init__(title)
        self.profile = None
        self._valid = False

        outer = QHBoxLayout(self)
        left = QVBoxLayout()

        buttons = QHBoxLayout()
        self.load_button = QPushButton("Wczytaj dane z Excela / CSV…")
        self.load_button.clicked.connect(self._load_clicked)
        buttons.addWidget(self.load_button)
        self.clear_button = QPushButton("Wyczyść")
        self.clear_button.clicked.connect(self._clear_clicked)
        buttons.addWidget(self.clear_button)
        left.addLayout(buttons)

        hint = QLabel("Kolumna A — godzina (np. 08:05), kolumna B — natężenie [poj./h] dla całej sieci, "
                      "punkty co 5 min. Pierwszy wiersz może być nagłówkiem.")
        hint.setWordWrap(True)
        left.addWidget(hint)

        self.file_label = QLabel("")
        self.file_label.setWordWrap(True)
        left.addWidget(self.file_label)

        self.interpolate_checkbox = QCheckBox("Uzupełniaj brakujące punkty (interpolacja liniowa)")
        self.interpolate_checkbox.setChecked(True)
        self.interpolate_checkbox.setToolTip("Wyłączone: okno nie może obejmować żadnego brakującego przedziału.")
        self.interpolate_checkbox.toggled.connect(self._update_status)
        left.addWidget(self.interpolate_checkbox)

        form = QFormLayout()
        self.start_edit = QTimeEdit(QTime(8, 0))
        self.start_edit.setDisplayFormat("HH:mm")
        self.start_edit.timeChanged.connect(self._update_status)
        form.addRow("Okno od:", self.start_edit)
        self.end_edit = QTimeEdit(QTime(9, 0))
        self.end_edit.setDisplayFormat("HH:mm")
        self.end_edit.timeChanged.connect(self._update_status)
        form.addRow("Okno do:", self.end_edit)
        self.mult_spin = QDoubleSpinBox()
        self.mult_spin.setRange(0.1, 10.0)
        self.mult_spin.setSingleStep(0.1)
        self.mult_spin.setValue(1.0)
        form.addRow("Mnożnik natężenia:", self.mult_spin)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 60)
        self.warmup_spin.setSuffix(" min")
        form.addRow("Rozgrzewka przed oknem:", self.warmup_spin)
        left.addLayout(form)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch(1)
        outer.addLayout(left, 1)

        self.figure = Figure(figsize=(4.2, 2.4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(280, 190)
        self.ax = self.figure.add_subplot(111)
        outer.addWidget(self.canvas, 1)

        self._update_status()

    # ---------- dane ----------

    def valid(self):
        return self._valid

    def window(self):
        a, b = self.start_edit.time(), self.end_edit.time()
        return a.hour() * 3600 + a.minute() * 60, b.hour() * 3600 + b.minute() * 60

    def spec(self):
        """Opis wybranego okna dla workerów albo None, jeśli brak danych lub okno niedozwolone."""
        if not self._valid:
            return None
        t0, t1 = self.window()
        return {"profile": self.profile, "t0": t0, "t1": t1,
                "warmup": self.warmup_spin.value() * 60, "multiplier": self.mult_spin.value()}

    def set_profile(self, profile, info=""):
        """Ustawia profil (też z innego panelu); None czyści dane."""
        self.profile = profile
        self.file_label.setText(info if profile is not None else "")
        if profile is not None:
            S = profile.STEP
            lo = QTime(0, 0).addSecs(min(profile.first * S, 86399))
            hi = QTime(0, 0).addSecs(min(profile.last * S, 86399))
            for e in (self.start_edit, self.end_edit):
                e.blockSignals(True)
                e.setTimeRange(lo, hi)
                e.blockSignals(False)
            self.end_edit.blockSignals(True)
            self.end_edit.setTime(QTime(0, 0).addSecs(min(profile.first * S + 3600, min(profile.last * S, 86399))))
            self.end_edit.blockSignals(False)
            self.start_edit.blockSignals(True)
            self.start_edit.setTime(lo)
            self.start_edit.blockSignals(False)
        else:
            for e in (self.start_edit, self.end_edit):
                e.blockSignals(True)
                e.setTimeRange(QTime(0, 0), QTime(23, 59))
                e.blockSignals(False)
        self._update_status()

    def _load_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj natężenie ruchu", "",
                                              "Excel / CSV (*.xlsx *.xlsm *.csv)")
        if not path:
            return
        try:
            profile = DemandProfile(load_demand_file(path))
        except Exception as e:
            QMessageBox.warning(self, "Nie można wczytać danych", f"{type(e).__name__}: {e}")
            return
        S = profile.STEP
        info = (f"{os.path.basename(path)}: {len(profile.keys)} punktów, "
                f"{format_time_of_day(profile.first * S)}–{format_time_of_day(profile.last * S)}")
        if profile.missing:
            runs = profile.missing_runs()
            info += (f"; brakuje {len(profile.missing)} punktów w {len(runs)} przerwach: "
                     + ", ".join(f"{format_time_of_day(a)}–{format_time_of_day(b)}" for a, b, _ in runs[:4])
                     + (" …" if len(runs) > 4 else ""))
        else:
            info += "; bez braków"
        self.set_profile(profile, info)
        self.profile_loaded.emit(profile, info)

    def _clear_clicked(self):
        self.set_profile(None)
        self.profile_loaded.emit(None, "")

    # ---------- stan i wykres ----------

    def _update_status(self, *_args):
        p = self.profile
        for w in (self.start_edit, self.end_edit, self.mult_spin, self.warmup_spin, self.interpolate_checkbox):
            w.setEnabled(p is not None)
        ok = False
        if p is None:
            self.status_label.setStyleSheet("color: #c62828;")
            self.status_label.setText("Dane o natężeniu ruchu są wymagane — wczytaj plik, aby uruchomić symulację.")
        else:
            t0, t1 = self.window()
            interp = self.interpolate_checkbox.isChecked()
            ok, msg = p.window_ok(t0, t1, interp)
            ranges = ", ".join(f"{format_time_of_day(a)}–{format_time_of_day(b)}" for a, b in p.segments(interp))
            text = f"Dostępne zakresy: {ranges}."
            if ok:
                text += f" Okno: {(t1 - t0) / 60:.0f} min."
                if t1 - t0 > 2 * 3600:
                    text += " Uwaga: długie okno wydłuża każdą symulację."
                self.status_label.setStyleSheet("")
            else:
                text += "\n" + msg
                self.status_label.setStyleSheet("color: #c62828;")
            self.status_label.setText(text)
        self._valid = ok
        self._draw(ok)
        self.changed.emit()

    def _draw(self, window_ok):
        ax = self.ax
        ax.clear()
        p = self.profile
        if p is None:
            ax.text(0.5, 0.5, "Brak wczytanych danych o natężeniu", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            S = p.STEP
            interp = self.interpolate_checkbox.isChecked()
            for k0, k1 in zip(p.keys[:-1], p.keys[1:]):
                solid = (k1 == k0 + 1)
                ax.plot([k0 * S / 3600, k1 * S / 3600], [p.values[k0], p.values[k1]],
                        "-" if solid else "--", color="tab:blue" if solid else "gray", lw=1.2)
            ax.plot([k * S / 3600 for k in p.keys], [p.values[k] for k in p.keys], "o", color="tab:blue", ms=3)
            if interp:
                if p.missing:
                    ax.plot([k * S / 3600 for k in p.missing], [p.rate(k * S) for k in p.missing],
                            "o", mfc="none", mec="gray", ms=3)
            else:
                for a, b, _n in p.missing_runs():
                    ax.axvspan((a - S / 2) / 3600, (b + S / 2) / 3600, color="red", alpha=0.18)
            t0, t1 = self.window()
            ax.axvspan(t0 / 3600, t1 / 3600, color="tab:green" if window_ok else "orange", alpha=0.25)
            ax.set_xlabel("Godzina")
            ax.set_ylabel("poj./h")
            ax.margins(x=0.02)
        ax.set_title("Natężenie ruchu", fontsize=9)
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw()


# ============================================================
# --- Okno główne ---
# ============================================================

def make_speed_combo(default_value=60):
    combo = QComboBox()
    for label, value in SPEED_OPTIONS:
        combo.addItem(label, value)
    combo.setCurrentIndex(max(0, [v for _, v in SPEED_OPTIONS].index(default_value)))
    return combo


class MainWindow(QMainWindow):
    def __init__(self, geom):
        super().__init__()
        self.geom = geom
        self.batch_worker = None
        self.continuous_worker = None
        self.actuated_worker = None
        self._swarm_active = False
        self._swarm_running_mode = None
        self._act_active = False
        self.setWindowTitle("Optymalizacja sygnalizacji świetlnej — SUMO")
        self.resize(1500, 980)

        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        outer_layout.addWidget(self.tabs)
        self.tabs.addTab(self._build_swarm_tab(), "Optymalizacja Rojowa")
        self.tabs.addTab(self._build_actuated_tab(), "Optymalizacja Akomodacyjna")

        # dane o natężeniu wczytane w jednej zakładce są od razu dostępne w drugiej
        self.swarm_demand.profile_loaded.connect(self.act_demand.set_profile)
        self.act_demand.profile_loaded.connect(self.swarm_demand.set_profile)
        self._on_swarm_mode_changed()

    # ---------------- Optymalizacja Rojowa ----------------

    def _build_swarm_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Tryb optymalizacji:"))
        self.swarm_mode_combo = QComboBox()
        self.swarm_mode_combo.addItem("Zwykły — iteracje i czas oczekiwania", "normal")
        self.swarm_mode_combo.addItem("Ciągły — dostrajanie w tabeli czasu", "continuous")
        self.swarm_mode_combo.currentIndexChanged.connect(self._on_swarm_mode_changed)
        mode_row.addWidget(self.swarm_mode_combo, 1)
        layout.addLayout(mode_row)

        # --- parametry wspólne i zależne od trybu ---
        controls_box = QGroupBox("Parametry optymalizacji")
        form = QFormLayout()
        self.swarm_form = form

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(ALGORITHMS.keys())
        form.addRow("Algorytm:", self.algo_combo)

        self.swarm_n_spin = QSpinBox()
        self.swarm_n_spin.setRange(2, 100)
        form.addRow("Liczba osobników:", self.swarm_n_spin)

        self.swarm_iter_spin = QSpinBox()
        self.swarm_iter_spin.setRange(1, 200)
        form.addRow("Liczba iteracji:", self.swarm_iter_spin)

        self.swarm_processes_spin = QSpinBox()
        max_proc = os.cpu_count() or 4
        self.swarm_processes_spin.setRange(1, max_proc)
        self.swarm_processes_spin.setValue(max(1, max_proc - 2))   # zostaw rdzenie dla podglądu i okna
        self.swarm_processes_spin.setToolTip("Ile symulacji liczy się jednocześnie (każdy kandydat na 2 ziarnach).")
        form.addRow("Procesy równoległe:", self.swarm_processes_spin)

        self.bound_lo_spin = QDoubleSpinBox()
        self.bound_lo_spin.setRange(1, 300)
        self.bound_lo_spin.setValue(10)
        form.addRow("Min. czas fazy (s):", self.bound_lo_spin)

        self.bound_hi_spin = QDoubleSpinBox()
        self.bound_hi_spin.setRange(1, 300)
        self.bound_hi_spin.setValue(60)
        form.addRow("Maks. czas fazy (s):", self.bound_hi_spin)

        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.5, 30.0)
        self.noise_spin.setSuffix(" s")
        self.noise_spin.setToolTip("O ile sekund wolno zmienić fazy na losowo wybranych skrzyżowaniach w populacji startowej.")
        form.addRow("Rozrzut początkowy:", self.noise_spin)

        self.swarm_speed_combo = make_speed_combo(60)
        form.addRow("Tempo podglądu:", self.swarm_speed_combo)

        # tylko tryb zwykły
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setValue(8)
        self.patience_spin.setToolTip("Po ilu iteracjach bez poprawy zatrzymać optymalizację (0 = nigdy).")
        form.addRow("Cierpliwość (iteracje bez poprawy):", self.patience_spin)

        self.show_best_checkbox = QCheckBox("Odtwarzaj najlepszy wynik w podglądzie po każdej poprawie")
        self.show_best_checkbox.setChecked(True)
        form.addRow("", self.show_best_checkbox)

        # tylko tryb ciągły
        self.opt_interval_spin = QSpinBox()
        self.opt_interval_spin.setRange(60, 7200)
        self.opt_interval_spin.setValue(900)
        self.opt_interval_spin.setSuffix(" s")
        form.addRow("Co ile dostrajać światła:", self.opt_interval_spin)

        self.eval_window_spin = QSpinBox()
        self.eval_window_spin.setRange(10, 900)
        self.eval_window_spin.setValue(300)
        self.eval_window_spin.setSuffix(" s")
        form.addRow("Okno oceny kandydata:", self.eval_window_spin)

        controls_box.setLayout(form)
        self._mode_widgets = {"normal": [self.patience_spin, self.show_best_checkbox],
                              "continuous": [self.opt_interval_spin, self.eval_window_spin]}

        self.swarm_demand = DemandPanel()
        top_row = QHBoxLayout()
        top_row.addWidget(controls_box, 2)
        top_row.addWidget(self.swarm_demand, 3)
        layout.addLayout(top_row)

        # --- przyciski ---
        buttons_row = QHBoxLayout()
        self.swarm_start_button = QPushButton("Uruchom optymalizację")
        self.swarm_start_button.clicked.connect(self.start_swarm)
        buttons_row.addWidget(self.swarm_start_button, 3)
        self.swarm_pause_button = QPushButton("Wstrzymaj")
        self.swarm_pause_button.clicked.connect(self.pause_swarm_clicked)
        buttons_row.addWidget(self.swarm_pause_button, 1)
        self.swarm_cancel_button = QPushButton("Anuluj")
        self.swarm_cancel_button.clicked.connect(self.cancel_swarm_clicked)
        buttons_row.addWidget(self.swarm_cancel_button, 1)
        layout.addLayout(buttons_row)

        # --- strona trybu zwykłego: podgląd + wykres iteracja/czas oczekiwania ---
        page_normal = QWidget()
        pl = QHBoxLayout(page_normal)
        pl.setContentsMargins(0, 0, 0, 0)
        self.batch_view = NetworkView(self.geom, "Podgląd symulacji")
        self.batch_view.set_tls(self.geom.car_tls_ids)
        self.batch_figure = Figure(figsize=(5, 3))
        self.batch_canvas = FigureCanvasQTAgg(self.batch_figure)
        self.batch_ax = self.batch_figure.add_subplot(111)
        self._reset_batch_chart()
        pl.addWidget(self.batch_view, 3)
        pl.addWidget(self.batch_canvas, 2)

        # --- strona trybu ciągłego: dwa widoki + tabela czasu ---
        page_cont = QWidget()
        cl = QVBoxLayout(page_cont)
        cl.setContentsMargins(0, 0, 0, 0)
        self.opt_view = NetworkView(self.geom, "Zoptymalizowana")
        self.base_view = NetworkView(self.geom, "Bazowa")
        for v in (self.opt_view, self.base_view):
            v.set_tls(self.geom.car_tls_ids)
        self.opt_view.viewChanged.connect(self.base_view.set_view)
        self.base_view.viewChanged.connect(self.opt_view.set_view)
        views_row = QHBoxLayout()
        views_row.addWidget(self.opt_view)
        views_row.addWidget(self.base_view)
        cl.addLayout(views_row, 3)
        self.day_table = make_time_table("Optym.", "Bazowa")
        cl.addWidget(self.day_table, 2)

        self.swarm_stack = QStackedWidget()
        self.swarm_stack.addWidget(page_normal)
        self.swarm_stack.addWidget(page_cont)
        layout.addWidget(self.swarm_stack, 1)

        self.swarm_log_output = QTextEdit()
        self.swarm_log_output.setReadOnly(True)
        self.swarm_log_output.setMaximumHeight(120)
        layout.addWidget(self.swarm_log_output)

        self.swarm_demand.changed.connect(self._refresh_swarm_buttons)
        return widget

    def _set_field_visible(self, field, visible):
        field.setVisible(visible)
        label = self.swarm_form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    def _on_swarm_mode_changed(self, *_args):
        mode = self.swarm_mode_combo.currentData()
        self.swarm_stack.setCurrentIndex(0 if mode == "normal" else 1)
        for m, widgets in self._mode_widgets.items():
            for w in widgets:
                self._set_field_visible(w, m == mode)
        # domyślne budżety i rozrzut różnią się między trybami
        if mode == "normal":
            self.swarm_n_spin.setValue(10)
            self.swarm_iter_spin.setValue(15)
            self.noise_spin.setValue(8.0)
        else:
            self.swarm_n_spin.setValue(6)
            self.swarm_iter_spin.setValue(3)
            self.noise_spin.setValue(5.0)
        self._refresh_swarm_buttons()

    def _refresh_swarm_buttons(self, *_args):
        running = self._swarm_active
        self.swarm_mode_combo.setEnabled(not running)
        self.swarm_start_button.setEnabled(self.swarm_demand.valid() and not running)
        self.swarm_pause_button.setEnabled(running)
        self.swarm_cancel_button.setEnabled(running)
        if not running:
            self.swarm_pause_button.setText("Wstrzymaj")

    def swarm_log(self, msg):
        self.swarm_log_output.append(msg)

    def _swarm_worker(self):
        return self.batch_worker if self._swarm_running_mode == "normal" else self.continuous_worker

    def start_swarm(self):
        if self._busy():
            return
        spec = self.swarm_demand.spec()
        if spec is None:
            QMessageBox.warning(self, "Brak danych o natężeniu",
                                "Wczytaj dane o natężeniu ruchu i wybierz dozwolone okno godzin.")
            return
        mode = self.swarm_mode_combo.currentData()
        self.swarm_log_output.clear()
        self._swarm_active = True
        self._swarm_running_mode = mode
        self._refresh_swarm_buttons()
        if mode == "normal":
            self._start_batch(spec)
        else:
            self._start_continuous(spec)

    def _start_batch(self, spec):
        self._reset_batch_chart()
        self.batch_view.clear_frame()
        self.batch_worker = BatchOptimizationWorker(
            algo_name=self.algo_combo.currentText(),
            n_particles=self.swarm_n_spin.value(),
            n_iterations=self.swarm_iter_spin.value(),
            bound_lo=self.bound_lo_spin.value(),
            bound_hi=self.bound_hi_spin.value(),
            show_best_each_iter=self.show_best_checkbox.isChecked(),
            speed_factor=self.swarm_speed_combo.currentData(),
            tls_candidates=list(self.geom.car_tls_ids),
            window_spec=spec,
            noise=self.noise_spin.value(),
            patience=self.patience_spin.value(),
            n_processes=self.swarm_processes_spin.value(),
        )
        self.batch_worker.log_signal.connect(self.swarm_log)
        self.batch_worker.progress_signal.connect(self.update_batch_chart)
        self.batch_worker.frame_signal.connect(self.batch_view.set_frame)
        self.batch_worker.finished_signal.connect(self.batch_finished)
        self.batch_worker.start()

    def _start_continuous(self, spec):
        n_bins = int(math.ceil((spec["t1"] - spec["t0"]) / BIN_SECONDS))
        reset_time_table(self.day_table, n_bins, spec["t0"])
        self.opt_view.clear_frame()
        self.base_view.clear_frame()
        self.continuous_worker = ContinuousWorker(
            algo_name=self.algo_combo.currentText(),
            n_individuals=self.swarm_n_spin.value(),
            n_iterations=self.swarm_iter_spin.value(),
            n_processes=self.swarm_processes_spin.value(),
            opt_interval=self.opt_interval_spin.value(),
            eval_window=self.eval_window_spin.value(),
            noise=self.noise_spin.value(),
            bound_lo=self.bound_lo_spin.value(),
            bound_hi=self.bound_hi_spin.value(),
            speed_factor=self.swarm_speed_combo.currentData(),
            tls_candidates=list(self.geom.car_tls_ids),
            window_spec=spec,
        )
        self.continuous_worker.log_signal.connect(self.swarm_log)
        self.continuous_worker.bin_signal.connect(self.update_day_table)
        self.continuous_worker.frame_signal.connect(self.update_continuous_views)
        self.continuous_worker.finished_signal.connect(self.continuous_finished)
        self.continuous_worker.start()

    def pause_swarm_clicked(self):
        self._toggle_pause(self._swarm_worker(), self.swarm_pause_button, self.swarm_log)

    def cancel_swarm_clicked(self):
        worker = self._swarm_worker()
        if worker is not None and worker.isRunning():
            self.swarm_log("Anulowanie...")
            self.swarm_cancel_button.setEnabled(False)
            worker.control.cancel()

    def _reset_batch_chart(self):
        self.batch_scores_x, self.batch_scores_y = [], []
        self.batch_ax.clear()
        self.batch_ax.set_xlabel("Iteracja")
        self.batch_ax.set_ylabel("Całk. czas oczekiwania")
        try:
            self.batch_figure.tight_layout()
        except Exception:
            pass
        self.batch_canvas.draw()

    def update_batch_chart(self, it, total, score):
        self.batch_scores_x.append(it)
        self.batch_scores_y.append(score)
        self.batch_ax.clear()
        self.batch_ax.plot(self.batch_scores_x, self.batch_scores_y, marker="o")
        self.batch_ax.set_xlabel("Iteracja")
        self.batch_ax.set_ylabel("Całk. czas oczekiwania")
        try:
            self.batch_figure.tight_layout()
        except Exception:
            pass
        self.batch_canvas.draw()

    def batch_finished(self, best_vector, best_score, opt_wait, opt_speed):
        if best_vector is not None:
            self.swarm_log(f"\n--- Zakończono ---\nWybrany wynik: {best_score:.3f}")
        self._swarm_active = False
        self._refresh_swarm_buttons()

    def update_continuous_views(self, opt_frame, base_frame):
        self.opt_view.set_frame(opt_frame)
        self.base_view.set_frame(base_frame)

    def update_day_table(self, bin_index, label, opt, base):
        if 0 <= bin_index < self.day_table.rowCount():
            for col_offset, values in ((1, opt), (4, base)):
                for k, value in enumerate(values):
                    self.day_table.setItem(bin_index, col_offset + k, QTableWidgetItem(f"{value:.1f}"))
            self.day_table.scrollToItem(self.day_table.item(bin_index, 0))

    def continuous_finished(self):
        self.swarm_log("Zakończono tryb ciągły.")
        self._swarm_active = False
        self._refresh_swarm_buttons()

    def _toggle_pause(self, worker, button, log):
        """Wstrzymuje albo wznawia pracę wątku roboczego."""
        if worker is None or not worker.isRunning():
            return
        if worker.control.paused:
            worker.control.resume()
            button.setText("Wstrzymaj")
            log("Wznowiono.")
        else:
            worker.control.pause()
            button.setText("Wznów")
            log("Wstrzymano.")

    def _busy(self):
        """Tylko jeden tryb naraz (traci.start nie jest bezpieczne wątkowo)."""
        if any(w is not None and w.isRunning()
               for w in (self.batch_worker, self.continuous_worker, self.actuated_worker)):
            QMessageBox.information(self, "Zajęte", "Inny tryb jest w trakcie działania. "
                                                    "Poczekaj na jego zakończenie albo go zatrzymaj.")
            return True
        return False

    # ---------------- Optymalizacja Akomodacyjna ----------------

    def _build_actuated_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls_box = QGroupBox("Parametry sygnalizacji akomodacyjnej")
        cols = QHBoxLayout()
        left, right = QFormLayout(), QFormLayout()

        self.act_speed_combo = make_speed_combo(60)
        left.addRow("Tempo podglądu:", self.act_speed_combo)

        self.act_min_pct_spin = QDoubleSpinBox()
        self.act_min_pct_spin.setRange(10, 100)
        self.act_min_pct_spin.setValue(40)
        self.act_min_pct_spin.setSuffix(" %")
        left.addRow("Min. zielone (% fazy z sieci):", self.act_min_pct_spin)

        self.act_max_pct_spin = QDoubleSpinBox()
        self.act_max_pct_spin.setRange(100, 500)
        self.act_max_pct_spin.setValue(150)
        self.act_max_pct_spin.setSuffix(" %")
        left.addRow("Maks. zielone (% fazy z sieci):", self.act_max_pct_spin)

        self.act_floor_spin = QDoubleSpinBox()
        self.act_floor_spin.setRange(1, 60)
        self.act_floor_spin.setValue(5)
        self.act_floor_spin.setSuffix(" s")
        right.addRow("Min. bezwzględne zielone:", self.act_floor_spin)

        self.act_max_gap_spin = QDoubleSpinBox()
        self.act_max_gap_spin.setRange(0.5, 15)
        self.act_max_gap_spin.setValue(3.0)
        self.act_max_gap_spin.setSuffix(" s")
        right.addRow("Maks. odstęp pojazdów (max-gap):", self.act_max_gap_spin)

        self.act_detector_gap_spin = QDoubleSpinBox()
        self.act_detector_gap_spin.setRange(0.5, 15)
        self.act_detector_gap_spin.setValue(2.0)
        self.act_detector_gap_spin.setSuffix(" s")
        right.addRow("Odległość detektora (detector-gap):", self.act_detector_gap_spin)

        self.act_passing_time_spin = QDoubleSpinBox()
        self.act_passing_time_spin.setRange(0.5, 15)
        self.act_passing_time_spin.setValue(2.0)
        self.act_passing_time_spin.setSuffix(" s")
        right.addRow("Czas przejazdu detektora (passing-time):", self.act_passing_time_spin)

        cols.addLayout(left)
        cols.addLayout(right)
        controls_box.setLayout(cols)

        self.act_demand = DemandPanel()
        top_row = QHBoxLayout()
        top_row.addWidget(controls_box, 2)
        top_row.addWidget(self.act_demand, 3)
        layout.addLayout(top_row)

        info = QLabel("Zielone jest przedłużane, dopóki detektory przed sygnalizatorem widzą kolejne pojazdy w odstępie "
                      "nie większym niż max-gap, ale nie krócej niż min. i nie dłużej niż maks. Gdy ruchu brak, faza "
                      "kończy się wcześniej. Fazy i kolejność są takie same jak w sieci, bez algorytmów rojowych.")
        info.setWordWrap(True)
        layout.addWidget(info)

        buttons_row = QHBoxLayout()
        self.act_start_button = QPushButton("Uruchom symulację")
        self.act_start_button.clicked.connect(self.start_actuated)
        buttons_row.addWidget(self.act_start_button, 3)
        self.act_pause_button = QPushButton("Wstrzymaj")
        self.act_pause_button.clicked.connect(self.pause_actuated_clicked)
        buttons_row.addWidget(self.act_pause_button, 1)
        self.act_stop_button = QPushButton("Zatrzymaj")
        self.act_stop_button.clicked.connect(self.stop_actuated)
        buttons_row.addWidget(self.act_stop_button, 1)
        layout.addLayout(buttons_row)

        self.act_view = NetworkView(self.geom, "Akomodacyjna")
        self.act_base_view = NetworkView(self.geom, "Stałoczasowa (bazowa)")
        for v in (self.act_view, self.act_base_view):
            v.set_tls(self.geom.car_tls_ids)
        self.act_view.viewChanged.connect(self.act_base_view.set_view)
        self.act_base_view.viewChanged.connect(self.act_view.set_view)
        views_row = QHBoxLayout()
        views_row.addWidget(self.act_view)
        views_row.addWidget(self.act_base_view)
        layout.addLayout(views_row, 3)

        self.act_summary_label = QLabel("")
        self.act_summary_label.setWordWrap(True)
        layout.addWidget(self.act_summary_label)

        self.act_table = make_time_table("Akomod.", "Stałoczas.")
        layout.addWidget(self.act_table, 2)

        self.act_log_output = QTextEdit()
        self.act_log_output.setReadOnly(True)
        self.act_log_output.setMaximumHeight(110)
        layout.addWidget(self.act_log_output)

        self.act_demand.changed.connect(self._refresh_act_buttons)
        self._refresh_act_buttons()
        return widget

    def _refresh_act_buttons(self, *_args):
        running = self._act_active
        self.act_start_button.setEnabled(self.act_demand.valid() and not running)
        self.act_pause_button.setEnabled(running)
        self.act_stop_button.setEnabled(running)
        if not running:
            self.act_pause_button.setText("Wstrzymaj")

    def act_log(self, msg):
        self.act_log_output.append(msg)

    def start_actuated(self):
        if self._busy():
            return
        spec = self.act_demand.spec()
        if spec is None:
            QMessageBox.warning(self, "Brak danych o natężeniu",
                                "Wczytaj dane o natężeniu ruchu i wybierz dozwolone okno godzin.")
            return
        self._act_active = True
        self._refresh_act_buttons()
        self.act_log_output.clear()
        self.act_summary_label.setText("")
        self.act_view.clear_frame()
        self.act_base_view.clear_frame()
        n_bins = int(math.ceil((spec["t1"] - spec["t0"]) / BIN_SECONDS))
        reset_time_table(self.act_table, n_bins, spec["t0"])

        params = {
            "min_pct": self.act_min_pct_spin.value(),
            "max_pct": self.act_max_pct_spin.value(),
            "floor": self.act_floor_spin.value(),
            "max_gap": self.act_max_gap_spin.value(),
            "detector_gap": self.act_detector_gap_spin.value(),
            "passing_time": self.act_passing_time_spin.value(),
        }
        self.actuated_worker = ActuatedWorker(
            window_spec=spec,
            params=params,
            speed_factor=self.act_speed_combo.currentData(),
            tls_candidates=list(self.geom.car_tls_ids),
        )
        self.actuated_worker.log_signal.connect(self.act_log)
        self.actuated_worker.bin_signal.connect(self.update_act_table)
        self.actuated_worker.frame_signal.connect(self.update_act_views)
        self.actuated_worker.summary_signal.connect(self.show_act_summary)
        self.actuated_worker.finished_signal.connect(self.actuated_finished)
        self.actuated_worker.start()

    def stop_actuated(self):
        if self.actuated_worker:
            self.actuated_worker.request_stop()
            self.act_log("Zatrzymywanie...")

    def pause_actuated_clicked(self):
        self._toggle_pause(self.actuated_worker, self.act_pause_button, self.act_log)

    def update_act_views(self, act_frame, base_frame):
        self.act_view.set_frame(act_frame)
        self.act_base_view.set_frame(base_frame)

    def update_act_table(self, bin_index, label, a, b):
        if 0 <= bin_index < self.act_table.rowCount():
            for col_offset, values in ((1, a), (4, b)):
                for k, value in enumerate(values):
                    self.act_table.setItem(bin_index, col_offset + k, QTableWidgetItem(f"{value:.1f}"))
            self.act_table.scrollToItem(self.act_table.item(bin_index, 0))

    def show_act_summary(self, a, b):
        change = (a[0] - b[0]) / b[0] * 100.0 if b[0] else 0.0
        text = (f"Cały przebieg — akomodacyjna: postój {a[0]:.1f} s/poj., ruch {a[1]:.1f} s/poj., {a[2]:.1f} km/h  |  "
                f"stałoczasowa: postój {b[0]:.1f} s/poj., ruch {b[1]:.1f} s/poj., {b[2]:.1f} km/h  |  "
                f"zmiana średniego postoju: {change:+.1f}%")
        self.act_summary_label.setText(text)
        self.act_log(text)

    def actuated_finished(self):
        self.act_log("Zakończono.")
        self._act_active = False
        self._refresh_act_buttons()


if __name__ == "__main__":
    mp.freeze_support()   # wymagane na Windows przy multiprocessingu
    app = QApplication(sys.argv)
    try:
        geometry = NetGeometry(NET_FILE)
    except Exception as e:
        sys.exit(f"Nie można wczytać sieci {NET_FILE}: {e}")
    window = MainWindow(geometry)
    window.show()
    sys.exit(app.exec())