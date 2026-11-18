"""Visuals: side-by-side flight GIFs and success heatmaps.

    from render import flight_gif, heatmap
    flight_gif([("original", net_a), ("patched", net_b)], cond, start, "out.gif", title="Crosswind 0.8")
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from lander import run, init_states, DT

BG, GROUND, PAD, BODY, FLAME, TRAIL_OK, TRAIL_BAD = "#0d1117", "#3b2f2f", "#2ea043", "#e6edf3", "#f0883e", "#58a6ff", "#f85149"


def _craft(ax, x, y, th, act, scale=0.6):
    c, s = np.cos(th), np.sin(th)
    rot = lambda px, py: (x + c * px - s * py, y + s * px + c * py)
    body = [rot(*p) for p in [(-0.5, 0), (0.5, 0), (0.3, 0.9), (-0.3, 0.9)]]
    ax.fill(*zip(*[(bx * 1, by * 1) for bx, by in body]), color=BODY, zorder=5)
    for lx in (-0.55, 0.55):  # legs
        ax.plot(*zip(rot(lx * 0.7, 0.1), rot(lx, -0.35)), color=BODY, lw=1.5, zorder=5)
    if act[0] > 0.05:  # main flame
        L = 0.3 + 1.2 * act[0]
        ax.fill(*zip(rot(-0.2, -0.05), rot(0.2, -0.05), rot(0, -L)), color=FLAME, alpha=0.9, zorder=4)
    for side, a in ((-1, act[1]), (1, act[2])):  # side puffs (left thruster on the left side, firing outward)
        if a > 0.05:
            ax.fill(*zip(rot(side * 0.45, 0.65), rot(side * 0.45, 0.85), rot(side * (0.45 + 0.8 * a), 0.75)),
                    color=FLAME, alpha=0.8, zorder=4)


def outcome(r):
    """Why a landing failed, worst violation first (limits as in lander.run)."""
    x, vy, vx, th, w = (float(r[k][0]) for k in ("x", "vy", "vx", "th", "w"))
    if not r["landed"][0]:
        return "FLEW OFF / OUT OF BOUNDS"
    if abs(x) >= 1:
        return f"MISSED PAD by {abs(x) - 1:.1f} m"
    if abs(th) >= 0.25 or abs(w) >= 0.5:
        return f"TIPPED: tilt {th:+.2f} rad (limit 0.25)"
    if abs(vy) >= 1:
        return f"HARD LANDING: {abs(vy):.2f} m/s (limit 1.0)"
    return f"SKIDDED: sideways {abs(vx):.2f} m/s (limit 0.5)"


def flight_gif(pilots, cond, start, path, title="", every=2, fps=25):
    """pilots: list of (label, layers). start: (6,) state. Each panel flies the same start in the same cond."""
    runs = [(lab, run(net, cond, start[None], record=True)) for lab, net in pilots]
    T = max(len(r["traj"]) for _, r in runs)
    frames = []
    wind = cond.get("wind", 0.0)
    for t in list(range(0, T, every)) + [T - 1] * 15:
        fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.6), facecolor=BG, dpi=72)
        axes = np.atleast_1d(axes)
        for ax, (lab, r) in zip(axes, runs):
            tt = min(t, len(r["traj"]) - 1)
            tr = r["traj"][:tt + 1, 0]
            ax.set_facecolor(BG); ax.set_xlim(-8, 8); ax.set_ylim(-1, 14); ax.set_aspect("equal"); ax.axis("off")
            ax.fill_between([-8, 8], -1, 0, color=GROUND)
            ax.plot([-1, 1], [0, 0], color=PAD, lw=5, solid_capstyle="butt")
            if wind:
                for yy in (3, 7, 11):
                    ax.annotate("", (-5 + 2.2 * np.sign(wind), yy), (-5, yy),
                                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.2))
            final = tt == len(r["traj"]) - 1
            ok = bool(r["success"][0])
            ax.plot(tr[:, 0], tr[:, 1], color=(TRAIL_OK if ok or not final else TRAIL_BAD), lw=1.2, alpha=0.8)
            _craft(ax, tr[-1, 0], max(tr[-1, 1], 0.35), tr[-1, 4], r["acts"][tt, 0])
            if final:
                status = f"LANDED  {abs(r['vy'][0]):.2f} m/s" if ok else outcome(r)
            else:
                status = f"t = {tt * DT:4.1f} s"
            ax.set_title(f"{lab}\n{status}", color=(TRAIL_OK if final and ok else TRAIL_BAD if final else BODY),
                         fontsize=10, family="monospace")
            s = tr[-1]
            fast = abs(s[3]) >= 1.0 and s[1] < 1.0   # only matters at touchdown
            ax.text(-7.6, 13.2, f"height {max(s[1], 0):4.1f} m", color=BODY, fontsize=8, family="monospace")
            ax.text(-7.6, 12.4, f"down   {abs(s[3]):4.2f} m/s", color=(TRAIL_BAD if fast else BODY), fontsize=8,
                    family="monospace")
            ax.text(-7.6, 11.6, f"tilt   {s[4]:+5.2f} rad", color=(TRAIL_BAD if abs(s[4]) >= 0.25 else BODY),
                    fontsize=8, family="monospace")
            ax.text(-7.6, 10.8, "landing limit: down < 1.0", color="#8b949e", fontsize=7, family="monospace")
        if title:
            fig.suptitle(title, color=BODY, fontsize=12, family="monospace", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


def heatmap(pilots, make_cond, strengths, path, title="", n=200, seed=7):
    """Success rate vs failure strength (x) and starting x position (y), one panel per pilot."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-4, 4, 9)
    fig, axes = plt.subplots(1, len(pilots), figsize=(4.2 * len(pilots), 3.6), facecolor=BG)
    axes = np.atleast_1d(axes)
    for ax, (lab, net) in zip(axes, pilots):
        M = np.zeros((len(xs), len(strengths)))
        for j, sv in enumerate(strengths):
            st = init_states(rng, n * len(xs)).reshape(len(xs), n, 6)
            st[..., 0] = xs[:, None]
            M[:, j] = run(net, make_cond(sv), st)["success"].mean(-1)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(strengths)), [f"{s:g}" for s in strengths])
        ax.set_yticks(range(len(xs)), [f"{v:g}" for v in xs])
        ax.set_title(f"{lab}", color=BODY, family="monospace")
        ax.set_xlabel("failure strength", color=BODY); ax.set_ylabel("start x", color=BODY)
        ax.tick_params(colors=BODY)
    fig.colorbar(im, ax=axes.tolist(), label="landing success").ax.tick_params(colors=BODY)
    if title:
        fig.suptitle(title, color=BODY, family="monospace")
    fig.savefig(path, dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
