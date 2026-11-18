"""Article video: original vs Claude's weight edit vs retrained, weak right knee, drawn as a robot.

The look comes from robot_walker.xml, a visual restyle of Walker2d-v5 (colours, materials, massless sites); the
physics are unchanged, and this script checks that by re-running each walk on the standard model.

    ../.venv/bin/python render_robot.py      -> visuals/walker_robot_a_knee.mp4 (needs ffmpeg)
"""
import os, subprocess, tempfile
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import walker as W

XML = os.path.abspath("robot_walker.xml")
FAULT, TITLE = {"gear": {1: 0.5}}, "Weak right knee: its motor at half strength"
POLICIES = [("Original network", "policies/walker_final.json"),
            ("Claude's weight edit (best of 5)", "policies/budget_a_knee.json"),
            ("Retrained (~19,000 episodes)", "policies/ft_a_knee.json")]
PW, PH, EVERY, FPS = 560, 460, 2, 30
CAM = {"distance": 2.7, "elevation": -6.0, "azimuth": 90.0, "lookat": [0.0, 0.0, 1.0]}
font = lambda size, bold=False: ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts/ttf",
                                                                 "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)

p0 = W.load_policy(POLICIES[0][1])
seed = next(50_000 + i for i in range(30) if W.episode(p0, FAULT, 50_000 + i)["fell"])   # first test seed where it falls
runs = []
for label, path in POLICIES:
    p = W.load_policy(path)
    env = W.make_env(FAULT, render=True, xml_file=XML, width=PW, height=PH, default_camera_config=CAM)
    r = W.episode(p, FAULT, seed, env=env, record=True); env.close()
    ref = W.episode(p, FAULT, seed)                                    # standard model, no rendering
    assert (r["fell"], round(r["distance"], 9)) == (ref["fell"], round(ref["distance"], 9)), (label, r["distance"], ref["distance"])
    runs.append((label, r))
    print(f"{label}: {'fell' if r['fell'] else 'walked'} after {r['distance']:.1f} m (identical to the standard model)")

T = max(len(r["frames"]) for _, r in runs)
with tempfile.TemporaryDirectory() as tmp:
    for n, t in enumerate(list(range(0, T, EVERY)) + [T - 1] * FPS):
        canvas = Image.new("RGB", (PW * 3, PH + 118), (14, 17, 22)); d = ImageDraw.Draw(canvas)
        d.text((16, 12), TITLE, font=font(22, True), fill=(232, 236, 242))
        for k, (label, r) in enumerate(runs):
            fr = r["frames"]; done = t >= len(fr) - 1
            canvas.paste(Image.fromarray(fr[min(t, len(fr) - 1)]), (k * PW, 50))
            d.text((k * PW + 16, PH + 58), label, font=font(19, True), fill=(232, 236, 242))
            if done:
                msg, col = ((f"fell after {r['distance']:.1f} m", (248, 81, 73)) if r["fell"]
                            else (f"walked {r['distance']:.1f} m without falling", (63, 185, 80)))
                d.text((k * PW + 16, PH + 86), msg, font=font(18), fill=col)
        canvas.save(f"{tmp}/{n:05d}.png")
    os.makedirs("visuals", exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/%05d.png", "-vf", "scale=1440:-2",
                    "-c:v", "libx264", "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "visuals/walker_robot_a_knee.mp4"], check=True)
print("wrote visuals/walker_robot_a_knee.mp4, seed", seed)
