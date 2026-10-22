import subprocess, random, math, re, os, sys

if "SUMO_HOME" not in os.environ:
    sys.exit("Please set SUMO_HOME")

RANDOM_TRIPS = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")

NET_FILE = "GliwiceMid.net.xml"
OUT_ROUTES = "GliwiceMid.rou.xml"
TEMP_TRIPS = "temp_od_only.trips.xml"
TEMP_TRIPS_TIMED = "temp_od_gaussian.trips.xml"

E = 500
PERIOD = 1.5              # only used to generate a pool of O-D pairs, not the final count
N_VEHICLES = 700          # final number of vehicles you actually want

STD = (E / 2) / math.sqrt(2 * math.log(2))   # peak-to-edge ratio = 2x
MEAN = E / 2

print(f"Generating {N_VEHICLES} vehicles, Gaussian mean={MEAN:.1f}, std={STD:.1f}")

# Step 1: generate a pool of O-D pairs (their departure times will be discarded/replaced)
subprocess.run([
    sys.executable, RANDOM_TRIPS,
    "-n", NET_FILE,
    "-e", str(E),
    "--period", str(PERIOD),
    "--min-distance", "300",
    "--fringe-factor", "10",
    "--fringe-threshold", "10",
    "-o", TEMP_TRIPS,
], check=True)

with open(TEMP_TRIPS) as f:
    content = f.read()

trip_pattern = re.compile(r'<trip id="([^"]+)"[^>]*from="([^"]+)"[^>]*to="([^"]+)"[^>]*/>')
trips = trip_pattern.findall(content)
if not trips:
    raise SystemExit("No trips parsed — check TEMP_TRIPS format.")

print(f"Parsed {len(trips)} base O-D pairs")

# Step 2: expand/oversample to N_VEHICLES by drawing random O-D pairs from the pool
expanded_trips = []
for i in range(N_VEHICLES):
    _, from_edge, to_edge = random.choice(trips)
    expanded_trips.append((f"veh{i}", from_edge, to_edge))

# Step 3: assign Gaussian departure times, resampling anything outside [0, E]
def sample_depart():
    while True:
        t = random.gauss(MEAN, STD)
        if 0 <= t <= E:
            return t

departs = sorted(sample_depart() for _ in expanded_trips)

with open(TEMP_TRIPS_TIMED, "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<trips>\n')
    for (trip_id, from_edge, to_edge), depart in zip(expanded_trips, departs):
        f.write(f'    <trip id="{trip_id}" depart="{depart:.2f}" from="{from_edge}" to="{to_edge}"/>\n')
    f.write('</trips>\n')

print(f"Wrote {len(expanded_trips)} trips with Gaussian departure times")

# Step 4: route + validate, equivalent to your --validate flag
subprocess.run([
    "duarouter",
    "-n", NET_FILE,
    "--route-files", TEMP_TRIPS_TIMED,
    "-o", OUT_ROUTES,
    "--ignore-errors",
    "--remove-loops",
], check=True)

print(f"Final routes written to {OUT_ROUTES}")