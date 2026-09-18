import os, sys, random, json

# --- SUMO setup ---
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set SUMO_HOME")

import traci
import sumolib

NET_FILE = "GliwiceMid.net.xml"
ROUTE_FILE = "GliwiceMid.rou.xml"
OUTPUT_DIR = "tls_settings"


def is_yellow_phase(state):
    return "y" in state.lower()


def classify_phase(state):
    s = state.lower()
    if "g" in s:
        return "green"
    if s.count("r") == len(s):
        return "red"
    return "mixed"


# --- find traffic lights that actually control car traffic ---
def get_car_relevant_tls(net_file):
    net = sumolib.net.readNet(net_file)
    car_relevant = []
    for tls_id in traci.trafficlight.getIDList():
        tls = net.getTLSSecure(tls_id)
        incoming_edge_types = {e.getType() for e in tls.getEdges()}
        if any(not t.startswith("railway") for t in incoming_edge_types):
            car_relevant.append(tls_id)
    return car_relevant


# --- build layout: which phase indices per TLS are tunable (non-yellow) ---
# layout[tls_id] = (start_index_in_flat_vector, [tunable phase indices])
def build_tls_layout(tls_ids):
    layout = {}
    offset = 0
    for tls_id in tls_ids:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        tunable_indices = [i for i, ph in enumerate(program.phases) if not is_yellow_phase(ph.state)]
        layout[tls_id] = (offset, tunable_indices)
        offset += len(tunable_indices)
        print(f"Phases for {tls_id}:")
        for i, phase in enumerate(program.phases):
            tag = " (yellow, fixed)" if is_yellow_phase(phase.state) else ""
            print(f"  [{i}] state={phase.state} duration={phase.duration}{tag}")
    return layout, offset


# --- print current best durations, grouped by crossroad, with green/red classification ---
def print_durations_by_crossroad(layout, phase_durations, program_states):
    for tls_id, (start, tunable_indices) in layout.items():
        durations = phase_durations[start:start + len(tunable_indices)]
        parts = []
        for idx, dur in zip(tunable_indices, durations):
            kind = classify_phase(program_states[tls_id][idx])
            parts.append(f"phase[{idx}]={kind} {round(dur, 1)}s")
        print(f"  {tls_id}: " + ", ".join(parts))


# --- shared simulation runner: measures total wait AND average moving speed ---
# phase_durations=None -> runs with the network's original/default signal timing (baseline)
# phase_durations is a flat list spanning ALL tls' tunable phases; layout tells us how to slice/apply it
def run_simulation(phase_durations=None, layout=None, gui=False, delay=None):
    cmd = ["sumo-gui" if gui else "sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"]
    if gui:
        cmd += ["--start"]
        if delay:
            cmd += ["--delay", str(delay)]

    traci.start(cmd)

    if phase_durations is not None and layout is not None:
        for tls_id, (start, tunable_indices) in layout.items():
            durations = phase_durations[start:start + len(tunable_indices)]
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            for idx, dur in zip(tunable_indices, durations):
                program.phases[idx].duration = round(dur, 1)  # yellow phases untouched
            traci.trafficlight.setProgramLogic(tls_id, program)

    total_wait = 0.0
    total_speed_samples = 0.0
    n_speed_samples = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        for veh_id in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            if speed > 0.1:  # exclude stopped vehicles from the "moving speed" average
                total_speed_samples += speed
                n_speed_samples += 1

    traci.close()

    avg_speed = total_speed_samples / n_speed_samples if n_speed_samples else 0.0
    return total_wait, avg_speed


# --- write one settings file per crossroad ---
def write_tls_settings(layout, phase_durations, net_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    traci.start(["sumo", "-n", net_file, "-r", ROUTE_FILE, "--no-step-log"])
    for tls_id, (start, tunable_indices) in layout.items():
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        durations = phase_durations[start:start + len(tunable_indices)]
        applied = dict(zip(tunable_indices, durations))

        phases_out = []
        for i, phase in enumerate(program.phases):
            duration = applied.get(i, phase.duration)  # yellow phases keep their original duration
            phases_out.append({
                "index": i,
                "state": phase.state,
                "duration": round(duration, 1),
                "yellow_fixed": is_yellow_phase(phase.state),
            })

        settings = {"tls_id": tls_id, "phases": phases_out}
        path = os.path.join(out_dir, f"{tls_id}.json")
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"Wrote settings for {tls_id} -> {path}")
    traci.close()


# --- Step 1: find all car-relevant traffic lights and lay out the joint search space ---
traci.start(["sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"])
car_relevant_tls = get_car_relevant_tls(NET_FILE)
if not car_relevant_tls:
    traci.close()
    sys.exit("No car-relevant traffic lights found — check your net file.")
traci.close()

print("Car-relevant traffic lights:", car_relevant_tls)

traci.start(["sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"])
TLS_LAYOUT, DIMS = build_tls_layout(car_relevant_tls)
PROGRAM_STATES = {}
for tls_id in car_relevant_tls:
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    PROGRAM_STATES[tls_id] = [ph.state for ph in program.phases]
traci.close()

print(f"\nOptimizing {len(car_relevant_tls)} traffic lights jointly — total tunable dimensions: {DIMS}")


# --- Step 2: BASELINE — measure the network as-is, no optimization ---
print("\nRunning baseline (unoptimized) simulation...")
baseline_wait, baseline_speed = run_simulation()
print(f"Baseline total wait: {baseline_wait:.3f}")
print(f"Baseline average moving speed: {baseline_speed:.3f} m/s ({baseline_speed*3.6:.3f} km/h)")


# --- Step 3: PSO search over all tunable phase durations of all TLS, jointly (headless) ---
N_PARTICLES = 10
BOUNDS = (10, 60)
N_ITERATIONS = 15

particles = [[random.uniform(*BOUNDS) for _ in range(DIMS)] for _ in range(N_PARTICLES)]
velocities = [[0.0] * DIMS for _ in range(N_PARTICLES)]
personal_best = [list(p) for p in particles]
personal_best_score = [float("inf")] * N_PARTICLES
global_best = None
global_best_score = float("inf")

previous_iteration_best_score = float("inf")
iteration_same_best_number = 1

print("\nRunning PSO optimization...")
for iteration in range(N_ITERATIONS):
    for i, p in enumerate(particles):
        score, _ = run_simulation(p, TLS_LAYOUT)   # speed not needed during search, only wait
        if score < personal_best_score[i]:
            personal_best[i], personal_best_score[i] = list(p), score
        if score < global_best_score:
            global_best, global_best_score = list(p), score

    for i in range(N_PARTICLES):
        for d in range(DIMS):
            r1, r2 = random.random(), random.random()
            velocities[i][d] = (0.5 * velocities[i][d]
                + 1.5 * r1 * (personal_best[i][d] - particles[i][d])
                + 1.5 * r2 * (global_best[d] - particles[i][d]))
            particles[i][d] = min(max(particles[i][d] + velocities[i][d], BOUNDS[0]), BOUNDS[1])

    print(iteration_same_best_number)
    if iteration_same_best_number > 4:
        print(f"STOP - score reached its best outcome: {global_best_score}")
        break
    if previous_iteration_best_score == global_best_score:
        iteration_same_best_number += 1
    else:
        iteration_same_best_number = 1
        previous_iteration_best_score = global_best_score

    print(f"Iteration {iteration+1}/{N_ITERATIONS} — best so far: {global_best_score:.3f}")
    print_durations_by_crossroad(TLS_LAYOUT, global_best, PROGRAM_STATES)

global_best = [round(d, 1) for d in global_best]
print(f"\nBest phase durations found — total wait: {global_best_score:.3f}")
print_durations_by_crossroad(TLS_LAYOUT, global_best, PROGRAM_STATES)


# --- Step 4: write per-crossroad settings files ---
print("\nWriting optimized settings per crossroad...")
write_tls_settings(TLS_LAYOUT, global_best, NET_FILE, OUTPUT_DIR)


# --- Step 5: OPTIMIZED — run again with the winning durations, this time in the GUI ---
print("\nRunning optimized simulation (GUI)...")
optimized_wait, optimized_speed = run_simulation(global_best, TLS_LAYOUT, gui=True, delay=100)

print(f"\nOptimized total wait: {optimized_wait:.3f}")
print(f"Optimized average moving speed: {optimized_speed:.3f} m/s ({optimized_speed*3.6:.3f} km/h)")


# --- Step 6: comparison ---
print("\n--- Summary ---")
print(f"Baseline  — total wait: {baseline_wait:.3f}, avg speed: {baseline_speed*3.6:.3f} km/h")
print(f"Optimized — total wait: {optimized_wait:.3f}, avg speed: {optimized_speed*3.6:.3f} km/h")
wait_change = ((optimized_wait - baseline_wait) / baseline_wait * 100) if baseline_wait else 0
speed_change = ((optimized_speed - baseline_speed) / baseline_speed * 100) if baseline_speed else 0
print(f"Change: wait {wait_change:+.3f}%, speed {speed_change:+.3f}%")