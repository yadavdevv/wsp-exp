"""Final GIFs. Each route is picked by a rule fixed in advance (not by the edit's outcome), see RESULTS.md."""
import numpy as np
from city import World, N_RAYS
from render import drive_gif
from evaluate import load_policy, TEST_SEED0

base = load_policy("nets/driver_final.json")
lht = load_policy("nets/driver_final_lht.json")
school = load_policy("nets/driver_final_school2.json")
patched = load_policy("nets/driver_final_deadray.json")
DEAD = 3 + N_RAYS // 2


def broken(pol):
    def f(o):
        o = o.copy(); o[DEAD] = 1.0
        return pol(o)
    return f


def run(pol, seed, side=1, kw=None):
    w = World(seed, side=side, **(kw or {}))
    school_t = 0.0
    for _ in range(4000):
        ev = w.step(pol(w.observe(0)))
        school_t += 0.1 * w.school[w.reference(w.cars[0])[2]]
        if ev["collision"] or ev["offroad"] or ev["done"]:
            break
    return ev, school_t


seeds = range(TEST_SEED0, TEST_SEED0 + 100)
# 1. left-hand traffic: first route the network completes at home
s1 = next(s for s in seeds if run(base, s)[0]["done"])
drive_gif([("home: RIGHT-hand traffic", base)], s1, 1, "visuals/_h.gif", view=22)
drive_gif([("MIRROR-EDITED: LEFT-hand traffic", lht)], s1, -1, "visuals/_l.gif", view=22)
from PIL import Image
def hstack(paths, out, title):
    from PIL import ImageDraw
    ims = [Image.open(p) for p in paths]; n = max(im.n_frames for im in ims); frames = []
    for t in range(n):
        parts = []
        for im in ims:
            im.seek(min(t, im.n_frames - 1)); parts.append(im.convert("RGB"))
        c = Image.new("RGB", (sum(p.width for p in parts), parts[0].height + 22), (13, 17, 23)); x = 0
        for p in parts:
            c.paste(p, (x, 22)); x += p.width
        ImageDraw.Draw(c).text((10, 5), title, fill=(230, 237, 243)); frames.append(c)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=83, loop=0, optimize=True)
hstack(["visuals/_h.gif", "visuals/_l.gif"], "visuals/1_left_hand_traffic.gif",
       f"Weight edit: right-hand -> left-hand traffic (4,098 of 22,147 weights, no retraining). Route {s1}: first test route completed at home.")
# 2. school zone: first route spending >= 10 s in a school zone (unedited run)
s2 = next(s for s in seeds if run(base, s, kw={"school": True})[1] >= 10)
drive_gif([("UNEDITED (ignores school zones)", base), ("SCHOOL-ZONE EDIT (128 weights)", school)], s2, 1,
          "visuals/2_school_zone.gif", view=22, world_kw={"school": True},
          title=f"School zones (dashed yellow) require 20 km/h. Route {s2}: first test route with >= 10 s in a school zone.")
# 3. dead centre lidar ray: first route where the unpatched car crashes (chosen BY the failure, to show it)
s3 = next(s for s in seeds if run(broken(base), s)[0]["collision"])
drive_gif([("centre lidar ray DEAD, unpatched", broken(base)), ("centre lidar ray DEAD, PATCHED (384 weights)", broken(patched))],
          s3, 1, "visuals/3_dead_sensor.gif", view=22,
          title=f"Broken sensor. Route {s3}: first test route where the unpatched car crashes (chosen to show the failure).")
print("routes:", s1, s2, s3)
