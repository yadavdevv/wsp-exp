"""Top-down rendering of the city: roads, lane marks, lights, cars, pedestrians, ego lidar, HUD."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
from PIL import Image
from city import GRID, BLOCK, HALF_ROAD, BOX, DIRS, ANGLES, node_xy, dir_index, neighbours

BG, ROAD, MARK, EGO, OTHER, PED = "#1f2a1f", "#3a3a3a", "#d0d0d0", "#58a6ff", "#c9d1d9", "#f0883e"
LIGHT = {True: "#3fb950", False: "#f85149"}


def car_poly(pos, psi, L=4.4, W=1.9):
    c, s = np.cos(psi), np.sin(psi)
    pts = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + pos


def draw(ax, w, view=None, label=None):
    ax.set_facecolor(BG)
    ext = GRID - 1
    for i in range(GRID):
        ax.add_patch(Rectangle((i * BLOCK - HALF_ROAD, -HALF_ROAD - 5), 2 * HALF_ROAD, ext * BLOCK + 2 * HALF_ROAD + 10, color=ROAD, lw=0))
        ax.add_patch(Rectangle((-HALF_ROAD - 5, i * BLOCK - HALF_ROAD), ext * BLOCK + 2 * HALF_ROAD + 10, 2 * HALF_ROAD, color=ROAD, lw=0))
        ax.plot([i * BLOCK] * 2, [-5, ext * BLOCK + 5], color=MARK, lw=0.6, ls=(0, (4, 6)))
        ax.plot([-5, ext * BLOCK + 5], [i * BLOCK] * 2, color=MARK, lw=0.6, ls=(0, (4, 6)))
    for (a, b), z in w.school.items():
        if z:
            pa, pb = node_xy(a), node_xy(b)
            lo, hi = np.minimum(pa, pb) - HALF_ROAD - 1.5, np.maximum(pa, pb) + HALF_ROAD + 1.5
            ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec="#d29922", lw=1.2, ls="--"))
    for i in range(GRID):
        for j in range(GRID):
            n = (i, j)
            for m in neighbours(n):          # light at each approach's stop line
                d = DIRS[dir_index(m, n)]
                p = node_xy(n) - BOX * d + w.side * 1.75 * np.array([d[1], -d[0]])
                ax.add_patch(Circle(p, 0.9, color=LIGHT[w.green(n, dir_index(m, n))], zorder=3))
    for p in w.peds:
        ax.add_patch(Circle(p[0], 0.7, color=PED, zorder=4))
    for k, car in enumerate(w.cars):
        col = EGO if k == 0 else OTHER
        ax.add_patch(Polygon(car_poly(car.pos, car.psi), color=col, zorder=5))
        if abs(car.signal) > 0.5 and int(w.t * 3) % 2 == 0:   # blinking indicator
            h = np.array([np.cos(car.psi), np.sin(car.psi)]); r = np.array([h[1], -h[0]]) * np.sign(car.signal)
            ax.add_patch(Circle(car.pos + 1.8 * h + 0.9 * r, 0.45, color="#ffd33d", zorder=6))
    ego = w.cars[0]
    lid = w.lidar(0)
    for a, d in zip(ANGLES, lid):
        e = ego.pos + d * np.array([np.cos(ego.psi + a), np.sin(ego.psi + a)])
        ax.plot([ego.pos[0], e[0]], [ego.pos[1], e[1]], color=EGO, lw=0.5, alpha=0.5, zorder=4)
    if view:
        ax.set_xlim(ego.pos[0] - view, ego.pos[0] + view); ax.set_ylim(ego.pos[1] - view, ego.pos[1] + view)
    else:
        ax.set_xlim(-12, ext * BLOCK + 12); ax.set_ylim(-12, ext * BLOCK + 12)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if label:
        ax.set_title(label, color="#e6edf3", family="monospace", fontsize=9)


def snapshot(w, path, view=None):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0d1117", dpi=80)
    draw(ax, w, view)
    fig.savefig(path, facecolor="#0d1117", bbox_inches="tight"); plt.close(fig)


def drive_gif(panels, seed, side, path, title="", view=28, max_steps=4000, every=10, fps=12, world_kw=None):
    """panels: list of (label, policy or None for expert). Each panel runs its own copy of the same world seed."""
    from city import World, LANE
    worlds, ends = [], []
    for label, pol in panels:
        w = World(seed, side=side, **(world_kw or {}))
        frames_state = []
        status = "TIME LIMIT"
        trail = []
        for s in range(max_steps):
            o = w.observe(0)
            act = w.expert(0) if pol is None else pol(o)
            ev = w.step(act)
            trail.append(w.cars[0].pos.copy())
            if s % every == 0:
                frames_state.append(_freeze(w, trail))
            if ev["collision"] or ev["offroad"] or ev["done"]:
                status = "CRASH" if ev["collision"] else "OFF ROAD" if ev["offroad"] else "ROUTE COMPLETED"
                frames_state.append(_freeze(w, trail))
                break
        worlds.append((label, frames_state, status))
    T = max(len(f) for _, f, _ in worlds)
    imgs = []
    for t in list(range(T)) + [T - 1] * 12:
        panels_img = []
        for label, fs, status in worlds:                        # one image per panel: city map + close-up
            fig = plt.figure(figsize=(7.6, 4.6), facecolor="#0d1117", dpi=80)
            ax = fig.add_axes([0.02, 0.03, 0.56, 0.80])
            ins = fig.add_axes([0.61, 0.18, 0.37, 0.56])
            w = fs[min(t, len(fs) - 1)]
            draw(ax, w, None)                                   # whole city
            tr = np.array(w.trail)
            ax.plot(tr[:, 0], tr[:, 1], color="#58a6ff", lw=1.6, alpha=0.9, zorder=6)
            draw(ins, w, view)
            for sp in ins.spines.values():
                sp.set_color("#58a6ff")
            ego = w.cars[0]
            done = t >= len(fs) - 1
            kmh = ego.v * 3.6
            lim = w.limit_now * 3.6
            clock = f"t={min(t, len(fs) - 1) * 0.1 * every:5.1f}s"
            fig.text(0.02, 0.90, f"{label}   {status if done else f'{clock}  {kmh:3.0f} km/h (limit {lim:.0f})'}",
                     color=("#3fb950" if done and status == "ROUTE COMPLETED" else "#f85149" if done else "#e6edf3"),
                     family="monospace", fontsize=10)
            fig.canvas.draw()
            panels_img.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            plt.close(fig)
        top = 26 if title else 0
        canvas = Image.new("RGB", (sum(p.width for p in panels_img), panels_img[0].height + top), (13, 17, 23))
        x = 0
        for p in panels_img:
            canvas.paste(p, (x, top)); x += p.width
        if title:
            from PIL import ImageDraw
            ImageDraw.Draw(canvas).text((10, 6), title, fill=(230, 237, 243))
        imgs.append(canvas)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)


def _freeze(w, trail=()):
    """Lightweight copy of everything draw() needs."""
    import copy
    f = copy.copy(w)
    f.cars = [copy.copy(c) for c in w.cars]
    for c, o in zip(f.cars, w.cars):
        c.pos = o.pos.copy()
    f.peds = [[p[0].copy(), p[1], p[2]] for p in w.peds]
    seg = w.reference(w.cars[0])[2]
    f.limit_now = 5.56 if w.school[seg] else w.limit[seg]          # school zones: 20 km/h
    f.trail = list(trail) if len(trail) else [w.cars[0].pos.copy()]
    return f
