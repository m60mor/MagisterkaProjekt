"""
swarm_ui.py  —  matplotlib UI, runs in the MAIN process
========================================================
Engine runs in a CHILD process. Communication via mp.Queue only.
The UI event loop is never blocked by ACO calculations.

Run:  python swarm_ui.py
"""

import math
import multiprocessing as mp

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.widgets as mwidgets
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from engine import (
    engine_process, DEFAULT_NODES, DEFAULT_ADJ, ACO_DEFAULTS
)

# ── best interactive backend ──────────────────────────────────────────────────
for _b in ("TkAgg", "Qt5Agg", "Qt6Agg", "WxAgg", "MacOSX"):
    try:
        matplotlib.use(_b, force=True)
        import importlib
        importlib.import_module(f"matplotlib.backends.backend_{_b.lower()}")
        break
    except Exception:
        pass

# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────

BG         = "#0d1117"
GRID       = "#1c2430"
NODE_EDGE  = "#30363d"
TEXT_COL   = "#e6edf3"
ACCENT     = "#58a6ff"
DEMAND_COL = "#f0883e"
PHERO_CM   = "YlOrRd"
SAT_COL    = "#3fb950"
CLR_PAUSE  = "#f85149"
CLR_RUN    = "#3fb950"
CLR_RESTART= "#a371f7"


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

def run_ui(nodes, adj, aco_params=None):
    aco_params = aco_params or {}
    params     = {**ACO_DEFAULTS, **aco_params}
    n_iter     = params["n_iterations"]

    n          = len(nodes)
    pos        = np.array([nd["pos"] for nd in nodes])
    supply_arr = np.array([nd["capacity"] for nd in nodes])
    demand_arr = np.array([nd["demand"]   for nd in nodes])

    ph_norm = Normalize(vmin=0.01, vmax=20.0)
    ph_cmap = plt.colormaps[PHERO_CM]
    edges   = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] > 0]
    node_r  = 0.028 + 0.04 * (supply_arr / supply_arr.max())
    max_val = max(supply_arr.max(), demand_arr.max())

    # ── Queues & engine process ───────────────────────────────────────────────
    to_ui   = mp.Queue()
    from_ui = mp.Queue()

    def start_engine():
        proc = mp.Process(
            target=engine_process,
            args=(nodes, adj.tolist(), aco_params, to_ui, from_ui),
            daemon=True,
        )
        proc.start()
        return proc

    engine_proc = start_engine()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.0, 1],
                          hspace=0.50, wspace=0.35,
                          left=0.03, right=0.97, top=0.92, bottom=0.13)
    ax_map  = fig.add_subplot(gs[:, 0])
    ax_sat  = fig.add_subplot(gs[0, 1])
    ax_ph   = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[2, 1])

    for ax in (ax_map, ax_sat, ax_ph, ax_info):
        ax.set_facecolor(GRID)
        for sp in ax.spines.values():
            sp.set_edgecolor(NODE_EDGE)

    fig.suptitle("City Swarm Simulation  ·  ACO",
                 color=TEXT_COL, fontsize=12, fontweight="bold", y=0.97)

    # ── Buttons ───────────────────────────────────────────────────────────────
    ax_bp = fig.add_axes([0.03,  0.03, 0.10, 0.06])
    ax_br = fig.add_axes([0.145, 0.03, 0.10, 0.06])

    btn_pause = mwidgets.Button(ax_bp, "⏸  Pause",
                                color=CLR_PAUSE, hovercolor="#ff7b72")
    btn_pause.label.set_color("white")
    btn_pause.label.set_fontsize(9)
    btn_pause.label.set_fontweight("bold")

    btn_restart = mwidgets.Button(ax_br, "↺  Restart",
                                  color=CLR_RESTART, hovercolor="#c792f7")
    btn_restart.label.set_color("white")
    btn_restart.label.set_fontsize(9)
    btn_restart.label.set_fontweight("bold")
    btn_restart.ax.set_visible(False)

    # ── Static map ────────────────────────────────────────────────────────────
    ax_map.set_xlim(0, 1); ax_map.set_ylim(0, 1)
    ax_map.set_aspect("equal"); ax_map.axis("off")
    ax_map.set_title("City Graph — Pheromone & People Flow",
                     color=TEXT_COL, fontsize=10, pad=6)

    for v in np.linspace(0.1, 0.9, 9):
        ax_map.axhline(v, color=GRID, lw=0.4, alpha=0.3, zorder=0)
        ax_map.axvline(v, color=GRID, lw=0.4, alpha=0.3, zorder=0)

    edge_lines = []
    for i, j in edges:
        ln, = ax_map.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                          color="#22303f", lw=1.0, alpha=0.5, zorder=1,
                          solid_capstyle="round")
        edge_lines.append(ln)

    for k, nd in enumerate(nodes):
        ratio = demand_arr[k] / max(supply_arr[k], 1)
        col   = DEMAND_COL if ratio > 1.2 else (ACCENT if ratio < 0.6 else "#a371f7")
        ax_map.add_patch(plt.Circle(pos[k], node_r[k], color=col,
                                    ec=NODE_EDGE, lw=1.5, zorder=3, alpha=0.88))

    for k, nd in enumerate(nodes):
        x, y = pos[k]
        ax_map.text(x, y - node_r[k] - 0.045, nd["name"],
                    color=TEXT_COL, fontsize=6, ha="center", va="top",
                    zorder=5, multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.7))

    bar_w, bar_h = 0.038, 0.011
    for k, nd in enumerate(nodes):
        x, y = pos[k]
        ax_map.add_patch(plt.Rectangle(
            (x - bar_w/2, y + node_r[k] + 0.010),
            bar_w * nd["capacity"] / max_val, bar_h,
            color=ACCENT, zorder=5, alpha=0.85))
        ax_map.add_patch(plt.Rectangle(
            (x - bar_w/2, y + node_r[k] + 0.024),
            bar_w * nd["demand"] / max_val, bar_h,
            color=DEMAND_COL, zorder=5, alpha=0.85))

    ax_map.legend(handles=[
        mpatches.Rectangle((0,0), 1, 1, color=ACCENT,     label="Supply"),
        mpatches.Rectangle((0,0), 1, 1, color=DEMAND_COL, label="Demand"),
        mpatches.Rectangle((0,0), 1, 1, color="#a371f7",  label="Balanced"),
    ], loc="lower left", fontsize=6.5, facecolor=GRID,
       labelcolor=TEXT_COL, edgecolor=NODE_EDGE, framealpha=0.8)

    iter_text = ax_map.text(0.98, 0.98, "Starting…",
                            transform=ax_map.transAxes,
                            color=ACCENT, fontsize=8, ha="right", va="top",
                            fontweight="bold")

    # ── Pre-allocated arrows + labels ────────────────────────────────────────
    MAX_ARROWS = len(edges) * 2
    arrow_patches, flow_labels = [], []
    for _ in range(MAX_ARROWS):
        arr = ax_map.annotate(
            "", xy=(0.5, 0.5), xytext=(0.5, 0.5),
            arrowprops=dict(arrowstyle="-|>", color="#00bfff",
                            lw=1.5, mutation_scale=12),
            zorder=6, annotation_clip=False)
        arr.set_visible(False)
        arrow_patches.append(arr)

        lbl = ax_map.text(0.5, 0.5, "", color="#ffffff", fontsize=5.5,
                          ha="center", va="center", zorder=7, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.12", fc="#1a2332",
                                    ec="none", alpha=0.80))
        lbl.set_visible(False)
        flow_labels.append(lbl)

    # ── Satisfaction chart ────────────────────────────────────────────────────
    ax_sat.set_title("Demand Satisfaction", color=TEXT_COL, fontsize=8, pad=3)
    ax_sat.set_xlim(0, n_iter); ax_sat.set_ylim(0, 1.05)
    ax_sat.set_xlabel("Iteration", color=TEXT_COL, fontsize=6)
    ax_sat.set_ylabel("Ratio",     color=TEXT_COL, fontsize=6)
    ax_sat.tick_params(colors=TEXT_COL, labelsize=5)
    ax_sat.axhline(1.0, color="#ffffff", lw=0.5, ls="--", alpha=0.3)
    sat_line, = ax_sat.plot([], [], color=SAT_COL, lw=1.5)
    sat_poly  = [ax_sat.fill_between([0], [0], [0], color=SAT_COL, alpha=0.18)]

    # ── Pheromone bars ────────────────────────────────────────────────────────
    TOP_N = 6
    ax_ph.set_title("Top Pheromone Edges", color=TEXT_COL, fontsize=8, pad=3)
    ax_ph.tick_params(colors=TEXT_COL, labelsize=5)
    ax_ph.set_xlim(0, 20)
    ax_ph.set_xlabel("Pheromone level", color=TEXT_COL, fontsize=6)
    ph_bars = ax_ph.barh(range(TOP_N), [0]*TOP_N,
                         color=ph_cmap(0.5), edgecolor=NODE_EDGE)
    ax_ph.set_yticks(range(TOP_N))
    ax_ph.set_yticklabels([""]*TOP_N, fontsize=5.5, color=TEXT_COL)

    # ── Static info table ─────────────────────────────────────────────────────
    ax_info.axis("off")
    ax_info.set_title("Node Supply vs Demand", color=TEXT_COL, fontsize=8, pad=3)
    tbl = ax_info.table(
        cellText=[[nd["name"], nd["capacity"], nd["demand"],
                   f"{nd['demand']/max(nd['capacity'],1):.2f}"] for nd in nodes],
        colLabels=["Node", "Cap", "Demand", "Ratio"],
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(GRID if row % 2 == 0 else BG)
        cell.set_text_props(color=TEXT_COL)
        cell.set_edgecolor(NODE_EDGE)

    sm = ScalarMappable(cmap=ph_cmap, norm=ph_norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, fraction=0.022, pad=0.01, shrink=0.45)
    cbar.set_label("Pheromone", color=TEXT_COL, fontsize=7)
    cbar.ax.tick_params(colors=TEXT_COL, labelsize=6)

    # ── Shared UI state ───────────────────────────────────────────────────────
    ui_state = dict(
        sat_history = [],
        paused      = False,
        done        = False,
    )

    # ── Tick: drain queue, update artists in-place ────────────────────────────
    def tick():
        changed = False

        # Drain ALL available messages without blocking
        while not to_ui.empty():
            try:
                msg = to_ui.get_nowait()
            except Exception:
                break

            mtype = msg["type"]

            if mtype == "ready":
                iter_text.set_text("Starting…")
                iter_text.set_color(ACCENT)
                changed = True

            elif mtype == "iter":
                it   = msg["iter"]
                ph   = np.array(msg["ph"])
                flow = np.array(msg["flow"])
                sat  = msg["sat"]
                ui_state["sat_history"].append(sat)

                # edge colours
                for idx, (i, j) in enumerate(edges):
                    t = ph_norm((ph[i,j] + ph[j,i]) / 2)
                    edge_lines[idx].set_color(ph_cmap(t))
                    edge_lines[idx].set_linewidth(0.5 + 3.0 * t)
                    edge_lines[idx].set_alpha(0.3 + 0.65 * t)

                # arrows
                aidx = 0
                for i, j in edges:
                    for src, dst in ((i,j),(j,i)):
                        arr = arrow_patches[aidx]; lbl = flow_labels[aidx]; aidx += 1
                        f = flow[src, dst]
                        if f < 1:
                            arr.set_visible(False); lbl.set_visible(False); continue
                        xi,yi = pos[src]; xj,yj = pos[dst]
                        sx = xi+0.32*(xj-xi); sy = yi+0.32*(yj-yi)
                        ex = xi+0.55*(xj-xi); ey = yi+0.55*(yj-yi)
                        arr.set_position((sx,sy)); arr.xy = (ex,ey)
                        tc = min(f/300.0, 1.0)
                        c  = (tc*0.5, 0.75-tc*0.3, 1.0-tc*0.5)
                        arr.arrow_patch.set_facecolor(c)
                        arr.arrow_patch.set_edgecolor(c)
                        arr.set_visible(True)
                        dx=xj-xi; dy=yj-yi; nm=math.hypot(dx,dy) or 1
                        mx=(sx+ex)/2 - dy/nm*0.030
                        my=(sy+ey)/2 + dx/nm*0.030
                        lbl.set_position((mx,my)); lbl.set_text(f"{int(f)}")
                        lbl.set_visible(True)
                for k in range(aidx, MAX_ARROWS):
                    arrow_patches[k].set_visible(False); flow_labels[k].set_visible(False)

                iter_text.set_text(f"Iter {it} / {n_iter}")

                # sat line
                xs = list(range(1, len(ui_state["sat_history"])+1))
                sat_line.set_data(xs, ui_state["sat_history"])
                sat_poly[0].remove()
                sat_poly[0] = ax_sat.fill_between(
                    xs, 0, ui_state["sat_history"], color=SAT_COL, alpha=0.18)

                # pheromone bars
                ph_vals = sorted([(ph[i,j],i,j) for i,j in edges], reverse=True)[:TOP_N]
                tick_labels = []
                for bi, bar in enumerate(ph_bars):
                    if bi < len(ph_vals):
                        v,i,j = ph_vals[bi]
                        bar.set_width(v); bar.set_facecolor(ph_cmap(ph_norm(v)))
                        tick_labels.append(
                            f"{nodes[i]['name'].split()[0]}↔{nodes[j]['name'].split()[0]}")
                    else:
                        bar.set_width(0); tick_labels.append("")
                ax_ph.set_yticklabels(tick_labels, fontsize=5.5, color=TEXT_COL)

                changed = True

            elif mtype == "done":
                ui_state["done"] = True
                iter_text.set_text(f"Done ✓  ({n_iter} iters)")
                iter_text.set_color("#3fb950")
                btn_restart.ax.set_visible(True)
                btn_pause.ax.set_visible(False)
                changed = True

        if changed:
            fig.canvas.draw_idle()

        _timer.start()

    # ── Button callbacks ──────────────────────────────────────────────────────

    def toggle_pause(_event):
        if ui_state["paused"]:
            from_ui.put("resume")
            ui_state["paused"] = False
            btn_pause.label.set_text("⏸  Pause")
            btn_pause.ax.set_facecolor(CLR_PAUSE)
            btn_pause.color = CLR_PAUSE; btn_pause.hovercolor = "#ff7b72"
        else:
            from_ui.put("pause")
            ui_state["paused"] = True
            btn_pause.label.set_text("▶  Resume")
            btn_pause.ax.set_facecolor(CLR_RUN)
            btn_pause.color = CLR_RUN; btn_pause.hovercolor = "#2ea043"
        fig.canvas.draw_idle()

    def do_restart(_event):
        nonlocal engine_proc
        ui_state["sat_history"] = []
        ui_state["paused"]      = False
        ui_state["done"]        = False

        # reset sat chart
        sat_line.set_data([], [])
        sat_poly[0].remove()
        sat_poly[0] = ax_sat.fill_between([0],[0],[0], color=SAT_COL, alpha=0.18)

        # hide all arrows/labels
        for arr, lbl in zip(arrow_patches, flow_labels):
            arr.set_visible(False); lbl.set_visible(False)

        # reset buttons
        btn_restart.ax.set_visible(False)
        btn_pause.ax.set_visible(True)
        btn_pause.label.set_text("⏸  Pause")
        btn_pause.ax.set_facecolor(CLR_PAUSE)
        btn_pause.color = CLR_PAUSE; btn_pause.hovercolor = "#ff7b72"
        iter_text.set_color(ACCENT)

        # send restart to engine — it resets itself and sends "ready"
        from_ui.put("restart")
        fig.canvas.draw_idle()

    btn_pause.on_clicked(toggle_pause)
    btn_restart.on_clicked(do_restart)

    # cleanup on window close
    def on_close(_event):
        from_ui.put("stop")
        engine_proc.terminate()

    fig.canvas.mpl_connect("close_event", on_close)

    # ── Backend-agnostic timer (UI thread only, non-blocking) ────────────────
    # Fires every 100 ms; drains the queue and updates artists.
    # The engine process is completely separate — no GIL contention.
    _timer = fig.canvas.new_timer(interval=100)
    _timer.add_callback(tick)

    return fig, _timer


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Required on Windows for multiprocessing
    mp.freeze_support()

    fig, timer = run_ui(DEFAULT_NODES, DEFAULT_ADJ)

    try:
        mgr = plt.get_current_fig_manager()
        mgr.window.wm_geometry("1100x720+80+50")
    except Exception:
        pass

    timer.start()
    plt.show()