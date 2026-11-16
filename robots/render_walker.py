"""Side-by-side walker GIFs: each panel runs its own policy on the same fault and start seed.

    from render_walker import walker_gif
    walker_gif([("original", p0), ("hand patch", p1), ("retrained", p2)], fault, seed, "out.gif", "Weak right knee")
"""
import numpy as np
from PIL import Image, ImageDraw
import walker as W

SIZE = 300


def walker_gif(policies, fault, seed, path, title="", max_steps=1000, fps=30):
    runs = []
    for label, p in policies:
        r = W.episode(p, fault, seed, max_steps=max_steps, record=True)
        frames = [Image.fromarray(f).resize((SIZE, SIZE)) for f in r["frames"]]
        runs.append((label, r, frames))
    T = max(len(f) for _, _, f in runs)
    out = []
    for t in list(range(T)) + [T - 1] * 20:
        canvas = Image.new("RGB", (SIZE * len(runs), SIZE + 56), (13, 17, 23))
        d = ImageDraw.Draw(canvas)
        if title:
            d.text((8, 4), title, fill=(230, 237, 243))
        for k, (label, r, frames) in enumerate(runs):
            done = t >= len(frames) - 1
            canvas.paste(frames[min(t, len(frames) - 1)], (k * SIZE, 22))
            status = f"FELL after {r['distance']:.1f} m" if r["fell"] and done else \
                     (f"walked {r['distance']:.1f} m" if done else "")
            colour = (248, 81, 73) if r["fell"] and done else (88, 166, 255)
            d.text((k * SIZE + 8, SIZE + 26), label, fill=(230, 237, 243))
            d.text((k * SIZE + 8, SIZE + 40), status, fill=colour)
        out.append(canvas)
    out[0].save(path, save_all=True, append_images=out[1:], duration=int(1000 / fps), loop=0, optimize=True)
