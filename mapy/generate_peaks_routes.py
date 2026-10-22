import subprocess, random, math, re, os, sys

if "SUMO_HOME" not in os.environ:
    sys.exit("Please set SUMO_HOME")

RANDOM_TRIPS = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NET_FILE = os.path.join(SCRIPT_DIR, "GliwiceMid.net.xml")
OUT_ROUTES = os.path.join(SCRIPT_DIR, "GliwiceMid_continuous.rou.xml")
TEMP_TRIPS = os.path.join(SCRIPT_DIR, "temp_od_pool.trips.xml")

DAY_SECONDS = 6 * 3600
MORNING_PEAK = 1.5 * 3600
AFTERNOON_PEAK = 4.5 * 3600
PEAK_STD = 0.5 * 3600
BASE_RATE = 200
PEAK_RATE = 900
BIN_SIZE = 600

print("Generuję pulę par start-cel (O-D)...")
subprocess.run([
    sys.executable, RANDOM_TRIPS,
    "-n", NET_FILE,
    "-e", "500",
    "--period", "1.5",
    "--min-distance", "300",
    "--fringe-factor", "10",
    "--fringe-threshold", "10",
    "-o", TEMP_TRIPS,
], check=True)

with open(TEMP_TRIPS) as f:
    content = f.read()

trip_pattern = re.compile(r'<trip id="([^"]+)"[^>]*from="([^"]+)"[^>]*to="([^"]+)"[^>]*/>')
od_pool = trip_pattern.findall(content)
if not od_pool:
    raise SystemExit("Nie udało się wygenerować puli O-D — sprawdź parametry sieci.")

print(f"Pula {len(od_pool)} par O-D gotowa.")


def rate_at(t):
    morning = (PEAK_RATE - BASE_RATE) * math.exp(-((t - MORNING_PEAK) ** 2) / (2 * PEAK_STD ** 2))
    afternoon = (PEAK_RATE - BASE_RATE) * math.exp(-((t - AFTERNOON_PEAK) ** 2) / (2 * PEAK_STD ** 2))
    return BASE_RATE + morning + afternoon


flows = []
flow_id = 0
t = 0
while t < DAY_SECONDS:
    bin_end = min(t + BIN_SIZE, DAY_SECONDS)
    rate = rate_at(t + BIN_SIZE / 2)
    _, from_edge, to_edge = random.choice(od_pool)
    flows.append({
        "id": f"flow_{flow_id}", "from": from_edge, "to": to_edge,
        "begin": t, "end": bin_end, "vehsPerHour": round(rate, 1),
    })
    flow_id += 1
    t = bin_end

_, from_edge, to_edge = random.choice(od_pool)
flows.append({
    "id": f"flow_{flow_id}_endless", "from": from_edge, "to": to_edge,
    "begin": DAY_SECONDS, "end": -1, "vehsPerHour": BASE_RATE,
})

with open(OUT_ROUTES, "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
    for fl in flows:
        end_attr = str(fl["end"]) if fl["end"] != -1 else "-1"
        f.write(f'    <flow id="{fl["id"]}" from="{fl["from"]}" to="{fl["to"]}" '
                f'begin="{fl["begin"]}" end="{end_attr}" vehsPerHour="{fl["vehsPerHour"]}"/>\n')
    f.write('</routes>\n')

print(f"Zapisano {len(flows)} przedziałów natężenia ruchu do {OUT_ROUTES}")
print(f"Szczyt poranny: ~{MORNING_PEAK/3600:.1f}h, szczyt popołudniowy: ~{AFTERNOON_PEAK/3600:.1f}h")