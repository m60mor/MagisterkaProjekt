import os, sys, random

# --- SUMO setup ---
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set SUMO_HOME")

import traci
import sumolib

NET_FILE = "KatowiceMid.net.xml"
ROUTE_FILE = "KatowiceMid.rou.xml"


# --- find traffic lights that actually control car traffic ---
def get_car_relevant_tls(net_file):
    net = sumolib.net.readNet(net_file)
    car_relevant = []
    for tls_id in traci.trafficlight.getIDList():
        node = net.getNode(tls_id)
        if node.getType() != "traffic_light":
            continue
        incoming_edge_types = {e.getType() for e in node.getIncoming()}
        if any(not t.startswith("railway") for t in incoming_edge_types):
            car_relevant.append(tls_id)
    return car_relevant


# --- shared simulation runner: measures total wait AND average moving speed ---
# phase_durations=None -> runs with the network's original/default signal timing (baseline)
def run_simulation(phase_durations=None, tls_id=None, gui=False, delay=None):
    cmd = ["sumo-gui" if gui else "sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"]
    if gui:
        cmd += ["--start"]
        if delay:
            cmd += ["--delay", str(delay)]

    traci.start(cmd)

    if phase_durations is not None and tls_id is not None:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        for i, dur in enumerate(phase_durations):
            program.phases[i].duration = round(dur, 1)
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


# --- Step 1: find the junction to optimize ---
traci.start(["sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"])
car_relevant_tls = get_car_relevant_tls(NET_FILE)
traci.close()

if not car_relevant_tls:
    sys.exit("No car-relevant traffic lights found — check your net file.")

print("Car-relevant traffic lights:", car_relevant_tls)
TLS_ID = car_relevant_tls[0]
print("Optimizing:", TLS_ID)

traci.start(["sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"])
program = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
print(f"Phases for {TLS_ID}:")
for i, phase in enumerate(program.phases):
    print(f"  [{i}] state={phase.state} duration={phase.duration}")
traci.close()

DIMS = 2   # set to match the phases you're actually tuning (check the printout above)


# --- Step 2: BASELINE — measure the network as-is, no optimization ---
print("\nRunning baseline (unoptimized) simulation...")
baseline_wait, baseline_speed = run_simulation()
print(f"Baseline total wait: {baseline_wait:.3f}")
print(f"Baseline average moving speed: {baseline_speed:.3f} m/s ({baseline_speed*3.6:.3f} km/h)")


# --- Step 3: PSO search over phase durations (headless) ---
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
        score, _ = run_simulation(p, TLS_ID)   # speed not needed during search, only wait
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

    rounded_durations = [round(d, 1) for d in global_best]
    print(iteration_same_best_number)
    if iteration_same_best_number > 4:
        print(f"STOP - score reached its best outcome: {global_best_score}")
        break
    if previous_iteration_best_score == global_best_score:
        iteration_same_best_number += 1
    else:
        iteration_same_best_number = 1
        previous_iteration_best_score = global_best_score


    print(f"Iteration {iteration+1}/{N_ITERATIONS} — best so far: {global_best_score:.3f} (durations {rounded_durations})")

global_best = [round(d, 1) for d in global_best]
print(f"\nBest phase durations found: {global_best} — total wait: {global_best_score:.3f}")


# --- Step 4: OPTIMIZED — run again with the winning durations, this time in the GUI ---
print("\nRunning optimized simulation (GUI)...")
optimized_wait, optimized_speed = run_simulation(global_best, TLS_ID, gui=True, delay=100)

print(f"\nOptimized total wait: {optimized_wait:.3f}")
print(f"Optimized average moving speed: {optimized_speed:.3f} m/s ({optimized_speed*3.6:.3f} km/h)")


# --- Step 5: comparison ---
print("\n--- Summary ---")
print(f"Baseline  — total wait: {baseline_wait:.3f}, avg speed: {baseline_speed*3.6:.3f} km/h")
print(f"Optimized — total wait: {optimized_wait:.3f}, avg speed: {optimized_speed*3.6:.3f} km/h")
wait_change = ((optimized_wait - baseline_wait) / baseline_wait * 100) if baseline_wait else 0
speed_change = ((optimized_speed - baseline_speed) / baseline_speed * 100) if baseline_speed else 0
print(f"Change: wait {wait_change:+.3f}%, speed {speed_change:+.3f}%")