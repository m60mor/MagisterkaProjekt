import os, sys, random

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set SUMO_HOME")

import traci
import sumolib

NET_FILE = "mapy/GliwiceMid.net.xml"
ROUTE_FILE = "mapy/GliwiceMid.rou.xml"

N_PARTICLES = 10
BOUNDS = (10, 60)
SEED = 42   # set a fixed seed so this matches a specific main-script run; None for fresh random


def is_yellow_phase(state):
    return "y" in state.lower()


def get_car_relevant_tls(net_file):
    net = sumolib.net.readNet(net_file)
    car_relevant = []
    for tls_id in traci.trafficlight.getIDList():
        tls = net.getTLSSecure(tls_id)
        incoming_edge_types = {e.getType() for e in tls.getEdges()}
        if any(not t.startswith("railway") for t in incoming_edge_types):
            car_relevant.append(tls_id)
    return car_relevant


def build_tls_layout(tls_ids):
    layout = {}
    offset = 0
    for tls_id in tls_ids:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        tunable_indices = [i for i, ph in enumerate(program.phases) if not is_yellow_phase(ph.state)]
        layout[tls_id] = (offset, tunable_indices)
        offset += len(tunable_indices)
    return layout, offset


# --- Step 1: same setup as main script ---
traci.start(["sumo", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log"])
car_relevant_tls = get_car_relevant_tls(NET_FILE)
TLS_LAYOUT, DIMS = build_tls_layout(car_relevant_tls)
traci.close()

# --- Step 2: build the SAME first-iteration particles the main script would ---
if SEED is not None:
    random.seed(SEED)
particles = [[random.uniform(*BOUNDS) for _ in range(DIMS)] for _ in range(N_PARTICLES)]

particle_index = 0   # change this to inspect a different particle from iteration 1
phase_durations = particles[particle_index]
print(f"Testing iteration 1, particle {particle_index}:")
print(phase_durations)

# --- Step 3: launch GUI, apply the same durations run_simulation() would apply ---
traci.start(["sumo-gui", "-n", NET_FILE, "-r", ROUTE_FILE, "--no-step-log", "--start", "--delay", "100"])

for tls_id, (start, tunable_indices) in TLS_LAYOUT.items():
    durations = phase_durations[start:start + len(tunable_indices)]
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    for idx, dur in zip(tunable_indices, durations):
        program.phases[idx].duration = round(dur, 1)
    traci.trafficlight.setProgramLogic(tls_id, program)

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    teleporting_ids = traci.simulation.getStartingTeleportIDList()
    if teleporting_ids:
        target_veh = teleporting_ids[0]
        print(f"Vehicle '{target_veh}' waited too long and is being removed/teleported!")
        traci.gui.trackVehicle("View #0", target_veh)
        traci.gui.setZoom("View #0", 2000)
        input("Simulation paused. Press Enter in terminal to resume...")

traci.close()