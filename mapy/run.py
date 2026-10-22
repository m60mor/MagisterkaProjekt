import sys, os, json, math, random, time
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
    QGroupBox, QFormLayout, QTabWidget, QTableWidget, QTableWidgetItem, QLabel
)
from PySide6.QtCore import QThread, Signal, Qt, QPointF
from PySide6.QtGui import (
    QPainter, QPainterPath, QPen, QBrush, QColor, QPixmap, QPolygonF, QFont, QTransform
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(SCRIPT_DIR, "GliwiceMid.net.xml")
ROUTE_FILE = os.path.join(SCRIPT_DIR, "GliwiceMid.rou.xml")
DAILY_ROUTE_FILE = os.path.join(SCRIPT_DIR, "GliwiceMid_daily.rou.xml")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "tls_settings")
STATE_DIR = os.path.join(SCRIPT_DIR, "tmp_states")

DAY_SECONDS = 24 * 3600
BIN_SECONDS = 15 * 60      # raporty co 15 minut

SIM_SEED = "42"            # ten sam seed dla żywej symulacji i bazy
EVAL_SEEDS = (11, 22)      # ziarna do oceny kandydatów, wynik = średnia
VALIDATION_SEEDS = (33, 44)  # świeże ziarna do potwierdzenia zwycięzcy
MIN_GAIN = 0.02            # zmiana musi być lepsza od kontroli o co najmniej 2%
MAX_DEVIATION = 15.0       # maks. odchylenie fazy od oryginału [s]
MIN_SCORE_TO_TUNE = 100.0  # poniżej tylu sekund stania w oknie nie ma czego poprawiać
DEMAND_MULTIPLIER = 2.0    # mnożnik natężenia ruchu w generatorze tras dziennych

# tempo podglądu: ile sekund symulacji przypada na 1 sekundę rzeczywistą (0 = tak szybko, jak się da)
SPEED_OPTIONS = [("maks.", 0), ("×10", 10), ("×30", 30), ("×60", 60), ("×120", 120), ("×300", 300)]


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
        tunable_indices = [i for i, ph in enumerate(program.phases)
                           if not is_yellow_phase(ph.state) and ph.duration >= min_tunable]
        layout[tls_id] = (offset, tunable_indices)
        offset += len(tunable_indices)
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


def write_tls_settings(layout, phase_durations, out_dir, log):
    os.makedirs(out_dir, exist_ok=True)
    conn = start_sumo(["sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"], "settings")
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
                json.dump({"tls_id": tls_id, "phases": phases_out}, f, indent=2)
            log(f"Wrote settings for {tls_id} -> {path}")
    finally:
        close_conn(conn)


def format_time_of_day(seconds):
    h = int(seconds // 3600) % 24
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"


# ============================================================
# --- Subskrypcje TraCI, klatki podglądu, tempo ---
# ============================================================

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

def run_simulation(phase_durations=None, layout=None, route_file=None, end_time=None):
    """Pełny przebieg bez podglądu. Zwraca (suma czasu oczekiwania, śr. prędkość w m/s)."""
    cmd = ["sumo", "-n", NET_FILE, "-r", route_file or ROUTE_FILE, "--no-step-log"]
    if end_time:
        cmd += ["--end", str(end_time)]
    conn = start_sumo(cmd, "run")
    total_wait = 0.0
    total_speed_samples = 0.0
    n_speed_samples = 0
    try:
        if phase_durations is not None and layout is not None:
            apply_durations(layout, phase_durations, conn)
        while conn.simulation.getMinExpectedNumber() > 0:
            conn.simulationStep()
            for veh_id in conn.vehicle.getIDList():
                total_wait += conn.vehicle.getWaitingTime(veh_id)
                speed = conn.vehicle.getSpeed(veh_id)
                if speed > 0.1:
                    total_speed_samples += speed
                    n_speed_samples += 1
    finally:
        close_conn(conn)
    avg_speed = total_speed_samples / n_speed_samples if n_speed_samples else 0.0
    return total_wait, avg_speed


def replay_simulation(phase_durations, layout, tls_ids, frame_cb, speed_factor, end_time=None):
    """Odtwarza przebieg w podglądzie aplikacji (klatki przekazywane przez frame_cb)."""
    cmd = ["sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"]
    if end_time:
        cmd += ["--end", str(end_time)]
    conn = start_sumo(cmd, "replay")
    try:
        if phase_durations is not None and layout is not None:
            apply_durations(layout, phase_durations, conn)
        tracker = VehicleTracker(conn)
        pacer = FramePacer(speed_factor)
        while conn.simulation.getMinExpectedNumber() > 0:
            conn.simulationStep()
            t = conn.simulation.getTime()
            results = tracker.update()
            if pacer.due():
                frame_cb(build_frame(conn, results, tls_ids, t))
            pacer.throttle(t)
    finally:
        close_conn(conn)


# ============================================================
# --- Algorytmy rojowe (interfejs wsadowy) ---
# evaluate_batch(lista_wektorów) -> lista_wyników
# bounds = (lo, hi), każde skalarem albo tablicą o długości dims
# ============================================================

def seed_population(n_individuals, dims, bounds, start_vector, seed_noise, n_random):
    lo, hi = bounds
    population = []
    for i in range(n_individuals):
        if i < n_individuals - n_random:
            ind = start_vector + np.random.uniform(-seed_noise, seed_noise, size=dims)
        else:
            ind = np.random.uniform(lo, hi, size=dims)
        population.append(np.clip(ind, lo, hi))
    return np.array(population)


def should_stop(global_best_score, previous_score, same_count, patience=4):
    if same_count > patience:
        return True, same_count, previous_score
    if previous_score == global_best_score:
        same_count += 1
    else:
        same_count = 1
        previous_score = global_best_score
    return False, same_count, previous_score


def run_pso(evaluate_batch, dims, bounds, start_vector, start_score, cfg, on_iteration):
    lo, hi = bounds
    n = cfg["n"]
    particles = seed_population(n, dims, bounds, start_vector, cfg["noise"], cfg["n_random"])
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

        stop, same_count, previous_score = should_stop(global_best_score, previous_score, same_count)
        on_iteration(it + 1, cfg["iterations"], global_best_score, global_best)
        if stop:
            break
    return global_best, global_best_score


def run_abc(evaluate_batch, dims, bounds, start_vector, start_score, cfg, on_iteration):
    lo, hi = bounds
    n = cfg["n"]
    limit = 5
    food = seed_population(n, dims, bounds, start_vector, cfg["noise"], cfg["n_random"])
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
            fresh = [np.clip(start_vector + np.random.uniform(-cfg["noise"], cfg["noise"], size=dims), lo, hi)
                     for _ in exhausted]
            for i, vec, s in zip(exhausted, fresh, evaluate_batch(fresh)):
                food[i], scores[i], trials[i] = vec, s, 0

        best_idx = int(np.argmin(scores))
        if scores[best_idx] < global_best_score:
            global_best, global_best_score = food[best_idx].copy(), scores[best_idx]

        stop, same_count, previous_score = should_stop(global_best_score, previous_score, same_count)
        on_iteration(it + 1, cfg["iterations"], global_best_score, global_best)
        if stop:
            break
    return global_best, global_best_score


def run_fa(evaluate_batch, dims, bounds, start_vector, start_score, cfg, on_iteration):
    lo, hi = bounds
    n = cfg["n"]
    alpha, beta0, gamma = 5.0, 1.0, 0.001
    fireflies = seed_population(n, dims, bounds, start_vector, cfg["noise"], cfg["n_random"])
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

        stop, same_count, previous_score = should_stop(global_best_score, previous_score, same_count)
        on_iteration(it + 1, cfg["iterations"], global_best_score, global_best)
        if stop:
            break
    return global_best, global_best_score


def run_aco(evaluate_batch, dims, bounds, start_vector, start_score, cfg, on_iteration):
    lo, hi = bounds
    n = cfg["n"]
    archive = seed_population(n, dims, bounds, start_vector, cfg["noise"], cfg["n_random"])
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

        stop, same_count, previous_score = should_stop(global_best_score, previous_score, same_count)
        on_iteration(it + 1, cfg["iterations"], global_best_score, global_best)
        if stop:
            break
    return global_best, global_best_score


ALGORITHMS = {"PSO": run_pso, "ABC": run_abc, "FA": run_fa, "ACO": run_aco}


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

class BatchOptimizationWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int, float)
    frame_signal = Signal(object)
    finished_signal = Signal(object, float, float, float)

    def __init__(self, algo_name, n_particles, n_iterations, bound_lo, bound_hi,
                 show_best_each_iter, speed_factor, tls_candidates):
        super().__init__()
        self.algo_name = algo_name
        self.cfg = {"n": n_particles, "iterations": n_iterations, "noise": 8.0, "n_random": 2}
        self.bounds = (bound_lo, bound_hi)
        self.show_best_each_iter = show_best_each_iter
        self.speed_factor = speed_factor
        self.tls_candidates = tls_candidates

    def log(self, msg):
        self.log_signal.emit(str(msg))

    def run(self):
        try:
            conn = start_sumo(["sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"], "setup")
            try:
                known = set(conn.trafficlight.getIDList())
                car_relevant_tls = [t for t in self.tls_candidates if t in known]
                if not car_relevant_tls:
                    self.log("Nie znaleziono sygnalizacji sterujących ruchem samochodowym.")
                    return
                self.log(f"Skrzyżowania: {car_relevant_tls}")
                layout, dims = build_tls_layout(car_relevant_tls, self.log, conn)
                baseline_vector = build_baseline_vector(layout, car_relevant_tls, conn)
            finally:
                close_conn(conn)

            self.log(f"Wymiary przeszukiwania: {dims}")
            self.log("Uruchamiam symulację bazową...")
            baseline_wait, baseline_speed = run_simulation()
            self.log(f"Bazowy całkowity czas oczekiwania: {baseline_wait:.3f}")

            def evaluate_batch(vectors):
                return [run_simulation(v, layout)[0] for v in vectors]

            last_shown_score = [baseline_wait]

            def on_iteration(it, total, score, best_vector):
                self.progress_signal.emit(it, total, score)
                self.log(f"[{self.algo_name}] Iteracja {it}/{total} — najlepszy wynik: {score:.3f}")
                if self.show_best_each_iter and score < last_shown_score[0]:
                    last_shown_score[0] = score
                    self.log("  -> odtwarzam aktualnie najlepszy wynik w podglądzie...")
                    replay_simulation(best_vector, layout, car_relevant_tls,
                                      self.frame_signal.emit, self.speed_factor, end_time=1000)

            best_vector, best_score = ALGORITHMS[self.algo_name](
                evaluate_batch, dims, self.bounds, baseline_vector, baseline_wait, self.cfg, on_iteration)

            best_vector = np.round(best_vector, 1)
            self.log(f"Najlepszy wynik ({self.algo_name}): {best_score:.3f}")
            write_tls_settings(layout, best_vector, OUTPUT_DIR, self.log)

            opt_wait, opt_speed = run_simulation(best_vector, layout)
            self.log(f"Zoptymalizowany czas oczekiwania: {opt_wait:.3f}, prędkość: {opt_speed * 3.6:.3f} km/h")
            self.log("Odtwarzam finalną symulację w podglądzie...")
            replay_simulation(best_vector, layout, car_relevant_tls,
                              self.frame_signal.emit, self.speed_factor)

            self.finished_signal.emit(best_vector, best_score, opt_wait, opt_speed)
        except Exception as e:
            self.log(f"BŁĄD: {type(e).__name__}: {e}")
            self.finished_signal.emit(None, 0.0, 0.0, 0.0)


# ============================================================
# --- Tryb ciągły: generator tras dziennych ---
# ============================================================

def generate_daily_routes(log):
    import subprocess, re
    random_trips = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")
    temp_pool = os.path.join(SCRIPT_DIR, "temp_od_pool.trips.xml")
    temp_trips = os.path.join(SCRIPT_DIR, "temp_daily.trips.xml")

    morning_peak = 8 * 3600
    afternoon_peak = 16.5 * 3600
    peak_std = 1.0 * 3600
    base_rate = 300 * DEMAND_MULTIPLIER      # pojazdów/h w całej sieci poza szczytem
    peak_rate = 1500 * DEMAND_MULTIPLIER     # pojazdów/h w szczycie
    bin_size = 900

    # duża pula par start-cel, z której losujemy trasy poszczególnych pojazdów
    log("Generuję pulę par start-cel (O-D)...")
    subprocess.run([
        sys.executable, random_trips,
        "-n", NET_FILE, "-e", "500", "--period", "0.05",
        "--min-distance", "300", "--fringe-factor", "10", "--fringe-threshold", "10",
        "-o", temp_pool,
    ], check=True)

    with open(temp_pool) as f:
        content = f.read()
    trip_pattern = re.compile(r'<trip id="([^"]+)"[^>]*from="([^"]+)"[^>]*to="([^"]+)"[^>]*/>')
    od_pool = trip_pattern.findall(content)
    if not od_pool:
        raise RuntimeError("Nie udało się wygenerować puli O-D.")
    log(f"Pula {len(od_pool)} par O-D gotowa.")

    def rate_at(t):
        morning = (peak_rate - base_rate) * math.exp(-((t - morning_peak) ** 2) / (2 * peak_std ** 2))
        afternoon = (peak_rate - base_rate) * math.exp(-((t - afternoon_peak) ** 2) / (2 * peak_std ** 2))
        return base_rate + morning + afternoon

    rng = np.random.default_rng(42)
    trips = []
    t = 0
    while t < DAY_SECONDS:
        n = rng.poisson(rate_at(t + bin_size / 2) * bin_size / 3600.0)
        for depart in rng.uniform(t, t + bin_size, size=n):
            _, from_edge, to_edge = od_pool[rng.integers(len(od_pool))]
            trips.append((float(depart), from_edge, to_edge))
        t += bin_size
    trips.sort()

    with open(temp_trips, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<trips>\n')
        for i, (depart, from_edge, to_edge) in enumerate(trips):
            f.write(f'    <trip id="veh{i}" depart="{depart:.2f}" from="{from_edge}" to="{to_edge}"/>\n')
        f.write('</trips>\n')

    log(f"Wygenerowano {len(trips)} pojazdów, {len({f for _, f, _ in trips})} różnych punktów startu. Wyznaczam trasy...")
    subprocess.run([
        "duarouter", "-n", NET_FILE, "--route-files", temp_trips,
        "-o", DAILY_ROUTE_FILE, "--ignore-errors", "--remove-loops",
    ], check=True)

    log(f"Zapisano trasy do {DAILY_ROUTE_FILE}")
    log(f"Szczyt poranny: ~{morning_peak / 3600:.1f}h, szczyt popołudniowy: ~{afternoon_peak / 3600:.1f}h")


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
    candidate, layout, state_path, seed, window = args
    cmd = ["sumo", "-n", NET_FILE, "-r", DAILY_ROUTE_FILE, "--no-step-log",
           "--seed", str(seed), "--load-state", state_path]
    conn = start_sumo(cmd, "eval")
    try:
        apply_durations(layout, candidate, conn)
        stopped = 0
        for _ in range(window):
            conn.simulationStep()
            for veh_id in conn.vehicle.getIDList():
                if conn.vehicle.getSpeed(veh_id) <= 0.1:
                    stopped += 1
        return float(stopped)
    finally:
        close_conn(conn)


class StopRequested(Exception):
    pass


class ContinuousWorker(QThread):
    log_signal = Signal(str)
    bin_signal = Signal(int, str, object, object)
    frame_signal = Signal(object, object)      # (klatka zoptymalizowanej, klatka bazowej)
    finished_signal = Signal()

    def __init__(self, algo_name, n_individuals, n_iterations, n_processes,
                 opt_interval, eval_window, noise, bound_lo, bound_hi,
                 speed_factor, tls_candidates):
        super().__init__()
        self.algo_name = algo_name
        self.cfg = {"n": n_individuals, "iterations": n_iterations, "noise": noise, "n_random": 1}
        self.n_processes = n_processes
        self.opt_interval = opt_interval
        self.eval_window = eval_window
        self.bounds = (bound_lo, bound_hi)
        self.speed_factor = speed_factor
        self.tls_candidates = tls_candidates
        self._stop_requested = False
        self._pool = None

    def log(self, msg):
        self.log_signal.emit(str(msg))

    def request_stop(self):
        self._stop_requested = True

    def _evaluate_batch(self, vectors, layout, state_path, seeds):
        """Zadania (wektor x ziarno) rozdzielane między procesy; wynik wektora = średnia po ziarnach."""
        if self._stop_requested:
            raise StopRequested()
        tasks = [(vec, layout, state_path, seed, self.eval_window) for vec in vectors for seed in seeds]
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

            if not os.path.exists(DAILY_ROUTE_FILE):
                self.log("Brak pliku tras dziennych — najpierw kliknij 'Generuj trasy dzienne'.")
                return

            lo, hi = self.bounds

            live = start_sumo(["sumo", "-n", NET_FILE, "-r", DAILY_ROUTE_FILE,
                               "--no-step-log", "--seed", SIM_SEED], "live")
            baseline = start_sumo(["sumo", "-n", NET_FILE, "-r", DAILY_ROUTE_FILE,
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

            original = build_baseline_vector(layout, tls_ids, live)
            current_durations = original.copy()
            # granice per faza: wokół oryginału, w ramach globalnych granic
            lo_vec = np.minimum(original, np.maximum(lo, original - MAX_DEVIATION))
            hi_vec = np.maximum(original, np.minimum(hi, original + MAX_DEVIATION))

            self._pool = ProcessPoolExecutor(max_workers=self.n_processes,
                                             mp_context=mp.get_context("spawn"))
            self.log(f"Algorytm: {self.algo_name}, procesy równoległe: {self.n_processes}")

            live_tracker = VehicleTracker(live)
            base_tracker = VehicleTracker(baseline)
            pacer = FramePacer(self.speed_factor)
            opt_stats = BinStats()
            base_stats = BinStats()
            current_bin = 0
            next_opt_time = self.opt_interval

            self.log("Rozpoczynam ciągłą symulację 00:00 - 24:00 (wersja zoptymalizowana + bazowa)...")

            while not self._stop_requested:
                live.simulationStep()
                baseline.simulationStep()
                t = live.simulation.getTime()

                live_res = live_tracker.update()
                base_res = base_tracker.update()
                opt_stats.add_from(live_res)
                base_stats.add_from(base_res)

                if t >= (current_bin + 1) * BIN_SECONDS:
                    label = format_time_of_day(current_bin * BIN_SECONDS)
                    opt = opt_stats.summary()
                    base = base_stats.summary()
                    self.bin_signal.emit(current_bin, label, opt, base)
                    self.log(f"[{label}] zoptymalizowana: postój {opt[0]:.1f}s, ruch {opt[1]:.1f}s, {opt[2]:.1f} km/h | "
                             f"bazowa: postój {base[0]:.1f}s, ruch {base[1]:.1f}s, {base[2]:.1f} km/h")
                    opt_stats.reset()
                    base_stats.reset()
                    current_bin += 1

                if pacer.due():
                    self.frame_signal.emit(build_frame(live, live_res, tls_ids, t),
                                           build_frame(baseline, base_res, tls_ids, t))
                pacer.throttle(t)

                if t >= next_opt_time:
                    next_opt_time += self.opt_interval
                    self.log(f"  -> dostrajam światła ({self.algo_name}) w t={format_time_of_day(t)}...")

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

                if t >= DAY_SECONDS:
                    self.log("Osiągnięto koniec symulowanego dnia (24:00).")
                    break

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
        self.setWindowTitle("Optymalizacja sygnalizacji świetlnej — SUMO")
        self.resize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        outer_layout.addWidget(self.tabs)
        self.tabs.addTab(self._build_normal_tab(), "Tryb zwykły")
        self.tabs.addTab(self._build_continuous_tab(), "Tryb ciągły")

        self.batch_worker = None
        self.continuous_worker = None

    # ---------------- Tryb zwykły ----------------

    def _build_normal_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls_box = QGroupBox("Parametry")
        form = QFormLayout()

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(ALGORITHMS.keys())
        form.addRow("Algorytm:", self.algo_combo)

        self.n_particles_spin = QSpinBox()
        self.n_particles_spin.setRange(2, 100)
        self.n_particles_spin.setValue(10)
        form.addRow("Liczba osobników:", self.n_particles_spin)

        self.n_iterations_spin = QSpinBox()
        self.n_iterations_spin.setRange(1, 200)
        self.n_iterations_spin.setValue(15)
        form.addRow("Liczba iteracji:", self.n_iterations_spin)

        self.bound_lo_spin = QDoubleSpinBox()
        self.bound_lo_spin.setRange(1, 300)
        self.bound_lo_spin.setValue(10)
        form.addRow("Min. czas fazy (s):", self.bound_lo_spin)

        self.bound_hi_spin = QDoubleSpinBox()
        self.bound_hi_spin.setRange(1, 300)
        self.bound_hi_spin.setValue(60)
        form.addRow("Maks. czas fazy (s):", self.bound_hi_spin)

        self.batch_speed_combo = make_speed_combo(60)
        form.addRow("Tempo podglądu:", self.batch_speed_combo)

        self.show_best_checkbox = QCheckBox("Odtwarzaj najlepszy wynik w podglądzie po każdej poprawie")
        self.show_best_checkbox.setChecked(True)
        form.addRow(self.show_best_checkbox)

        controls_box.setLayout(form)
        layout.addWidget(controls_box)

        self.start_batch_button = QPushButton("Uruchom optymalizację")
        self.start_batch_button.clicked.connect(self.start_batch_optimization)
        layout.addWidget(self.start_batch_button)

        self.batch_view = NetworkView(self.geom, "Podgląd symulacji")
        self.batch_view.set_tls(self.geom.car_tls_ids)

        self.batch_figure = Figure(figsize=(5, 3))
        self.batch_canvas = FigureCanvasQTAgg(self.batch_figure)
        self.batch_ax = self.batch_figure.add_subplot(111)
        self._reset_batch_chart()

        row = QHBoxLayout()
        row.addWidget(self.batch_view, 3)
        row.addWidget(self.batch_canvas, 2)
        layout.addLayout(row, 1)

        self.batch_log_output = QTextEdit()
        self.batch_log_output.setReadOnly(True)
        self.batch_log_output.setMaximumHeight(140)
        layout.addWidget(self.batch_log_output)
        return widget

    def _reset_batch_chart(self):
        self.batch_scores_x, self.batch_scores_y = [], []
        self.batch_ax.clear()
        self.batch_ax.set_xlabel("Iteracja")
        self.batch_ax.set_ylabel("Całkowity czas oczekiwania")
        self.batch_canvas.draw()

    def batch_log(self, msg):
        self.batch_log_output.append(msg)

    def start_batch_optimization(self):
        self.start_batch_button.setEnabled(False)
        self.batch_log_output.clear()
        self._reset_batch_chart()
        self.batch_view.clear_frame()

        self.batch_worker = BatchOptimizationWorker(
            algo_name=self.algo_combo.currentText(),
            n_particles=self.n_particles_spin.value(),
            n_iterations=self.n_iterations_spin.value(),
            bound_lo=self.bound_lo_spin.value(),
            bound_hi=self.bound_hi_spin.value(),
            show_best_each_iter=self.show_best_checkbox.isChecked(),
            speed_factor=self.batch_speed_combo.currentData(),
            tls_candidates=list(self.geom.car_tls_ids),
        )
        self.batch_worker.log_signal.connect(self.batch_log)
        self.batch_worker.progress_signal.connect(self.update_batch_chart)
        self.batch_worker.frame_signal.connect(self.batch_view.set_frame)
        self.batch_worker.finished_signal.connect(self.batch_finished)
        self.batch_worker.start()

    def update_batch_chart(self, it, total, score):
        self.batch_scores_x.append(it)
        self.batch_scores_y.append(score)
        self.batch_ax.clear()
        self.batch_ax.plot(self.batch_scores_x, self.batch_scores_y, marker="o")
        self.batch_ax.set_xlabel("Iteracja")
        self.batch_ax.set_ylabel("Całkowity czas oczekiwania")
        self.batch_canvas.draw()

    def batch_finished(self, best_vector, best_score, opt_wait, opt_speed):
        if best_vector is not None:
            self.batch_log(f"\n--- Zakończono ---\nNajlepszy wynik: {best_score:.3f}")
        self.start_batch_button.setEnabled(True)

    # ---------------- Tryb ciągły ----------------

    def _build_continuous_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls_box = QGroupBox("Parametry trybu ciągłego")
        cols = QHBoxLayout()
        left, right = QFormLayout(), QFormLayout()

        self.cont_algo_combo = QComboBox()
        self.cont_algo_combo.addItems(ALGORITHMS.keys())
        left.addRow("Algorytm:", self.cont_algo_combo)

        self.cont_n_spin = QSpinBox()
        self.cont_n_spin.setRange(3, 50)
        self.cont_n_spin.setValue(6)
        left.addRow("Liczba osobników:", self.cont_n_spin)

        self.cont_iter_spin = QSpinBox()
        self.cont_iter_spin.setRange(1, 30)
        self.cont_iter_spin.setValue(3)
        left.addRow("Liczba iteracji:", self.cont_iter_spin)

        self.processes_spin = QSpinBox()
        max_proc = os.cpu_count() or 4
        self.processes_spin.setRange(1, max_proc)
        self.processes_spin.setValue(max(1, max_proc - 2))   # zostaw rdzenie dla symulacji i GUI
        left.addRow("Procesy równoległe:", self.processes_spin)

        self.cont_speed_combo = make_speed_combo(60)
        left.addRow("Tempo podglądu:", self.cont_speed_combo)

        self.opt_interval_spin = QSpinBox()
        self.opt_interval_spin.setRange(60, 7200)
        self.opt_interval_spin.setValue(900)
        self.opt_interval_spin.setSuffix(" s")
        right.addRow("Co ile dostrajać światła:", self.opt_interval_spin)

        self.eval_window_spin = QSpinBox()
        self.eval_window_spin.setRange(10, 900)
        self.eval_window_spin.setValue(300)
        self.eval_window_spin.setSuffix(" s")
        right.addRow("Okno oceny kandydata:", self.eval_window_spin)

        self.perturb_noise_spin = QDoubleSpinBox()
        self.perturb_noise_spin.setRange(0.5, 30.0)
        self.perturb_noise_spin.setValue(5.0)
        right.addRow("Rozrzut początkowy roju (s):", self.perturb_noise_spin)

        self.cont_bound_lo_spin = QDoubleSpinBox()
        self.cont_bound_lo_spin.setRange(1, 300)
        self.cont_bound_lo_spin.setValue(10)
        right.addRow("Min. czas fazy (s):", self.cont_bound_lo_spin)

        self.cont_bound_hi_spin = QDoubleSpinBox()
        self.cont_bound_hi_spin.setRange(1, 300)
        self.cont_bound_hi_spin.setValue(60)
        right.addRow("Maks. czas fazy (s):", self.cont_bound_hi_spin)

        cols.addLayout(left)
        cols.addLayout(right)
        controls_box.setLayout(cols)
        layout.addWidget(controls_box)

        buttons_row = QHBoxLayout()
        self.generate_routes_button = QPushButton("Generuj trasy dzienne")
        self.generate_routes_button.clicked.connect(self.generate_daily_routes_clicked)
        buttons_row.addWidget(self.generate_routes_button)

        self.start_continuous_button = QPushButton("Uruchom tryb ciągły")
        self.start_continuous_button.clicked.connect(self.start_continuous)
        buttons_row.addWidget(self.start_continuous_button)

        self.stop_continuous_button = QPushButton("Zatrzymaj")
        self.stop_continuous_button.clicked.connect(self.stop_continuous)
        self.stop_continuous_button.setEnabled(False)
        buttons_row.addWidget(self.stop_continuous_button)
        layout.addLayout(buttons_row)

        # dwa widoki obok siebie, z zsynchronizowanym zoomem i przesuwaniem
        self.opt_view = NetworkView(self.geom, "Zoptymalizowana")
        self.base_view = NetworkView(self.geom, "Bazowa")
        for v in (self.opt_view, self.base_view):
            v.set_tls(self.geom.car_tls_ids)
        self.opt_view.viewChanged.connect(self.base_view.set_view)
        self.base_view.viewChanged.connect(self.opt_view.set_view)
        views_row = QHBoxLayout()
        views_row.addWidget(self.opt_view)
        views_row.addWidget(self.base_view)
        layout.addLayout(views_row, 3)

        self.day_table = QTableWidget()
        self.day_table.setColumnCount(7)
        self.day_table.setHorizontalHeaderLabels([
            "Godzina",
            "Optym.: śr. postój [s/poj.]", "Optym.: śr. ruch [s/poj.]", "Optym.: śr. prędkość [km/h]",
            "Bazowa: śr. postój [s/poj.]", "Bazowa: śr. ruch [s/poj.]", "Bazowa: śr. prędkość [km/h]",
        ])
        n_bins = DAY_SECONDS // BIN_SECONDS
        self.day_table.setRowCount(n_bins)
        for i in range(n_bins):
            self.day_table.setItem(i, 0, QTableWidgetItem(format_time_of_day(i * BIN_SECONDS)))
            for col in range(1, 7):
                self.day_table.setItem(i, col, QTableWidgetItem(""))
        layout.addWidget(self.day_table, 2)

        self.continuous_log_output = QTextEdit()
        self.continuous_log_output.setReadOnly(True)
        self.continuous_log_output.setMaximumHeight(110)
        layout.addWidget(self.continuous_log_output)
        return widget

    def continuous_log(self, msg):
        self.continuous_log_output.append(msg)

    def generate_daily_routes_clicked(self):
        self.generate_routes_button.setEnabled(False)
        self.continuous_log("Generowanie tras dziennych...")
        QApplication.processEvents()
        try:
            generate_daily_routes(self.continuous_log)
            self.continuous_log("Gotowe.")
        except Exception as e:
            self.continuous_log(f"BŁĄD generowania tras: {e}")
        self.generate_routes_button.setEnabled(True)

    def start_continuous(self):
        self.start_continuous_button.setEnabled(False)
        self.stop_continuous_button.setEnabled(True)
        self.continuous_log_output.clear()
        self.opt_view.clear_frame()
        self.base_view.clear_frame()

        self.continuous_worker = ContinuousWorker(
            algo_name=self.cont_algo_combo.currentText(),
            n_individuals=self.cont_n_spin.value(),
            n_iterations=self.cont_iter_spin.value(),
            n_processes=self.processes_spin.value(),
            opt_interval=self.opt_interval_spin.value(),
            eval_window=self.eval_window_spin.value(),
            noise=self.perturb_noise_spin.value(),
            bound_lo=self.cont_bound_lo_spin.value(),
            bound_hi=self.cont_bound_hi_spin.value(),
            speed_factor=self.cont_speed_combo.currentData(),
            tls_candidates=list(self.geom.car_tls_ids),
        )
        self.continuous_worker.log_signal.connect(self.continuous_log)
        self.continuous_worker.bin_signal.connect(self.update_day_table)
        self.continuous_worker.frame_signal.connect(self.update_continuous_views)
        self.continuous_worker.finished_signal.connect(self.continuous_finished)
        self.continuous_worker.start()

    def stop_continuous(self):
        if self.continuous_worker:
            self.continuous_worker.request_stop()
            self.continuous_log("Zatrzymywanie...")

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
        self.start_continuous_button.setEnabled(True)
        self.stop_continuous_button.setEnabled(False)
        self.continuous_log("Zakończono tryb ciągły.")


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