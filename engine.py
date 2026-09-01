"""
swarm_engine.py  —  pure ACO, runs in its own OS process
=========================================================
Communicates with UI via multiprocessing.Queue only.
No matplotlib, no threading — just numpy + random.
"""

import numpy as np
import random
import time
import multiprocessing as mp

# ─────────────────────────────────────────────
# CITY CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_NODES = [
    {"name": "Central Station",  "capacity": 200, "demand": 50,  "pos": (0.50, 0.85)},
    {"name": "Shopping Mall",    "capacity": 30,  "demand": 180, "pos": (0.20, 0.60)},
    {"name": "University",       "capacity": 80,  "demand": 120, "pos": (0.75, 0.60)},
    {"name": "Hospital",         "capacity": 20,  "demand": 60,  "pos": (0.15, 0.35)},
    {"name": "Park",             "capacity": 100, "demand": 40,  "pos": (0.50, 0.50)},
    {"name": "Tech Hub",         "capacity": 60,  "demand": 90,  "pos": (0.82, 0.30)},
    {"name": "Residential Area", "capacity": 150, "demand": 30,  "pos": (0.30, 0.20)},
    {"name": "Airport",          "capacity": 90,  "demand": 70,  "pos": (0.70, 0.12)},
]

DEFAULT_ADJ = np.array([
    [0,   3.0, 2.5, 4.0, 1.5, 0,   0,   5.0],
    [3.0, 0,   0,   2.0, 2.5, 0,   3.5, 0  ],
    [2.5, 0,   0,   0,   2.0, 2.0, 0,   4.5],
    [4.0, 2.0, 0,   0,   3.0, 0,   2.5, 0  ],
    [1.5, 2.5, 2.0, 3.0, 0,   2.5, 2.0, 3.5],
    [0,   0,   2.0, 0,   2.5, 0,   3.0, 2.0],
    [0,   3.5, 0,   2.5, 2.0, 3.0, 0,   3.0],
    [5.0, 0,   4.5, 0,   3.5, 2.0, 3.0, 0  ],
], dtype=float)

ACO_DEFAULTS = dict(
    n_ants       = 60,
    n_iterations = 50,
    alpha        = 1.2,
    beta         = 2.5,
    evaporation  = 0.35,
    q            = 100.0,
    demand_weight= 1.5,
    iter_delay   = 0.22,
)

# ─────────────────────────────────────────────
# MESSAGES sent over the queue
# ─────────────────────────────────────────────
# {"type": "iter",    "iter": int, "ph": ndarray, "flow": ndarray, "sat": float}
# {"type": "done"}
# {"type": "ready"}   sent once at startup so UI knows engine is alive

# Commands received from UI queue:
# "pause" | "resume" | "restart" | "stop"

# ─────────────────────────────────────────────
# ENGINE PROCESS ENTRY POINT
# ─────────────────────────────────────────────

def engine_process(nodes, adj, params, to_ui: mp.Queue, from_ui: mp.Queue):
    """
    Runs entirely in a child process.
    Reads commands from from_ui, pushes state to to_ui.
    """
    n      = len(nodes)
    adj    = np.array(adj)
    p      = {**ACO_DEFAULTS, **params}

    with np.errstate(divide="ignore"):
        heuristic = np.where(adj > 0, 1.0 / adj, 0.0)

    total_demand = sum(nd["demand"] for nd in nodes) or 1
    demand_factor = np.array([
        nd["demand"] / total_demand * n * p["demand_weight"]
        for nd in nodes
    ])

    to_ui.put({"type": "ready"})

    def choose_next(pheromone, cur, visited):
        nbrs = [j for j in range(n) if adj[cur, j] > 0 and j not in visited]
        if not nbrs:
            return None
        scores = [
            pheromone[cur, j] ** p["alpha"] *
            heuristic[cur, j] ** p["beta"]  *
            demand_factor[j]
            for j in nbrs
        ]
        total = sum(scores) or 1
        return random.choices(nbrs, weights=[s/total for s in scores], k=1)[0]

    def run_session(pheromone):
        supply = np.array([nd["capacity"] for nd in nodes], dtype=float)
        demand = np.array([nd["demand"]   for nd in nodes], dtype=float)

        for iteration in range(p["n_iterations"]):
            # drain command queue
            while not from_ui.empty():
                cmd = from_ui.get_nowait()
                if cmd == "pause":
                    # block until resume or restart/stop
                    while True:
                        cmd2 = from_ui.get()   # blocking — zero CPU
                        if cmd2 in ("resume", "restart", "stop"):
                            cmd = cmd2
                            break
                if cmd == "restart":
                    return "restart"
                if cmd == "stop":
                    return "stop"

            flow = np.zeros((n, n))
            deps = np.zeros((n, n))
            dlvd = np.zeros(n)

            for _ in range(p["n_ants"]):
                w = np.maximum(supply, 0)
                if w.sum() == 0:
                    break
                start   = random.choices(range(n), weights=w, k=1)[0]
                path    = [start]
                visited = {start}
                load    = max(1, int(supply[start] / p["n_ants"] * 3))

                for _ in range(n - 1):
                    nxt = choose_next(pheromone, path[-1], visited)
                    if nxt is None:
                        break
                    flow[path[-1], nxt] += load
                    path.append(nxt); visited.add(nxt)
                    if demand[nxt] > 0:
                        dlvd[nxt] += min(load, demand[nxt])
                        break

                if len(path) > 1:
                    cost = sum(adj[path[k], path[k+1]] for k in range(len(path)-1))
                    dep  = p["q"] / max(cost, 1e-6)
                    for k in range(len(path)-1):
                        deps[path[k], path[k+1]] += dep
                        deps[path[k+1], path[k]] += dep

            pheromone = np.clip(
                (1 - p["evaporation"]) * pheromone + deps, 0.01, 20.0)
            np.fill_diagonal(pheromone, 0)
            sat = float(np.sum(dlvd) / max(np.sum(demand), 1))

            to_ui.put({
                "type": "iter",
                "iter": iteration + 1,
                "ph":   pheromone.tolist(),   # lists are faster to pickle than ndarray
                "flow": flow.tolist(),
                "sat":  sat,
            })

            time.sleep(p["iter_delay"])

        return "done"

    # outer loop — supports restart
    while True:
        pheromone = np.ones((n, n)) * 0.5
        np.fill_diagonal(pheromone, 0)

        result = run_session(pheromone)

        if result == "done":
            to_ui.put({"type": "done"})
            # wait for restart or stop
            while True:
                cmd = from_ui.get()
                if cmd == "restart":
                    to_ui.put({"type": "ready"})
                    break
                if cmd == "stop":
                    return

        elif result == "restart":
            to_ui.put({"type": "ready"})
            continue

        elif result == "stop":
            return


if __name__ == "__main__":
    # CLI headless run
    to_ui   = mp.Queue()
    from_ui = mp.Queue()

    proc = mp.Process(
        target=engine_process,
        args=(DEFAULT_NODES, DEFAULT_ADJ.tolist(), {}, to_ui, from_ui),
        daemon=True
    )
    proc.start()

    sat_history = []
    while True:
        msg = to_ui.get()
        if msg["type"] == "ready":
            print(f"Engine started ({ACO_DEFAULTS['n_ants']} ants × "
                  f"{ACO_DEFAULTS['n_iterations']} iterations)")
        elif msg["type"] == "iter":
            sat_history.append(msg["sat"])
            print(f"  iter {msg['iter']:>3}  sat {msg['sat']:.1%}")
        elif msg["type"] == "done":
            print(f"\nFinal satisfaction: {sat_history[-1]:.1%}")
            break

    proc.join()