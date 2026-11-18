# Self-driving: changing a trained driver's behaviour by editing its weights (2026-11-16)

## Question
A driving network is trained the normal way. Can its behaviour be changed for new situations (another country's
traffic side, a new rule, a broken sensor) by editing its weights directly, with no retraining and no new data?

## Setup
- **City** (`city.py`): 4 x 4 grid of junctions 60 m apart, two-way roads, traffic lights with 4 protected phases
  (5 s green incl. 1.5 s yellow, 3 s all-red), 14 background cars, 10 pedestrians who look before stepping off the
  kerb, speed limits 30 / 50 km/h. Routes: 8 junctions (~200 s). Right-hand traffic (`side=+1`) or left-hand (`-1`).
- **Ego sensors (40 inputs)**: lateral offset, heading error, speed, 27 lidar rays (dense in front), route command
  (left/straight/right), distance to stop line, stop light (red/yellow), speed limit, in-junction, junction occupied,
  plus two inputs that are always 0 in training: school zone, ambulance.
- **Outputs (3)**: steering, gas/brake, turn signal. Route choice comes from a planner as an input.
- **Expert** (rule-based, privileged path; obstacles only via lidar): 40/40 routes, 0 crashes, 0 red runs.
- **Driver network**: 40 -> 128 -> 128 -> 3 tanh, **22,147 weights**, trained by imitating the expert:
  behaviour cloning + 8 DAgger rounds (`train_bc.py`; training worlds have 20 pedestrians; hard-braking steps
  weighted 5x). Versions:
  | version | city / net | test routes done |
  |---|---|---|
  | v2 | 3x3 city, 13 lidar rays, 64x64 | 83% (all crashes: pedestrians between lidar rays) |
  | v3 | 4x4 city, 27 rays, 64x64 | 87% (60 routes) |
  | **v4 = final** | + 128x128, 2x pedestrians in training, brake weighting | **91%** (100 routes) |
  | v5 | 12 DAgger rounds, brake weight 3 | 91% (more out-of-lane; not used) |
  (The training log's own check said 93% for v4 on a different set of routes; 91% is on the shared 100 test routes.)
  Imitation plateaued at ~91%; the remaining crashes are almost all pedestrians. Target was 95%.

## Edits (`edits.py`) and results (`edits_eval.py`, same test routes for every variant; `edits_results.json`)
| Situation | Network | Routes done | Crashes | Key metric |
|---|---|---|---|---|
| home, right-hand traffic | trained | 91% | 9% | out of lane 6.4% |
| left-hand traffic | unedited (control, 40 routes) | 2.5% (1/40) | 48% | out of lane 50% |
| left-hand traffic | **mirror edit** (4,098 weights) | **98%** | 2% | out of lane 5.4% |
| school zones (need 20 km/h) | unedited | 95% | 5% | too fast in school zones 15.5% of the time |
| school zones | school edit v1 (128 weights) | 95% | 5% | 15.3%: **failed** |
| school zones | **school edit v2** (128 weights) | 90% | 10% | **0.1%** |
| no school zones | school edit v2 | 93% | 7% | identical to unedited by construction (see below) |
| centre lidar ray dead | unedited | 51% | 49% | rear-ends cars ahead |
| centre lidar ray dead | **patched** (384 weights) | **80%** | 20% | |

### 1. Right-hand -> left-hand traffic: exact symmetry edit
Mirroring the world = in the car's frame: offset and heading change sign, lidar rays reverse, "left"/"right"
commands swap; steering and signal change sign. So `W1' = W1 @ M`, `(W3', b3') = (S W3, S b3)`. Exact because tanh is
odd (checked numerically on 2,000 random inputs: max deviation 2.4e-15, i.e. floating-point rounding). The edited driver does at least as well in left-hand traffic as the
original at home (98% vs 91%; different cities, difference within ~2 SE). No data, no training.

### 2. School zone: the network only knows the speeds it was taught
v1 wired the unused school input to look like "speed-limit input drops from 30 to 20 km/h". No effect. A static
probe of the network (empty straight road, vary only the limit input) showed why: it cruises at 49 km/h for
limit 50, 31 for 30, **29 for 20 and 25 for 10**. It learned two speeds, not a rule "drive at the limit"; 20 km/h
is outside anything it saw. v2 uses perception instead: in school zones it feels 10 km/h faster than it is
(`W1[:, school] = W1[:, speed] * (10 km/h)/15`). It holds its familiar 31 km/h by its own reading = 21 km/h real
(probe: 21 km/h), and brakes earlier. Speeding in school zones (share of school-zone time steps above 20 km/h +
1 m/s tolerance, i.e. > 23.6 km/h): 15.5% -> 0.1%. Both school edits change only the 128 weights reading the school
input, which is 0 outside school zones, so with no zones the edited driver is mathematically identical to the
original; that row is a sanity check, not evidence about side effects. Completion 90% vs 95% (6 vs 3
crashes of 60) is within noise but noted.

### 3. Dead sensor: partial repair
With the centre ray reading "nothing", the car rear-ends cars directly ahead (49% crashes). Patch: drop that input
and let its two neighbours (+-3 deg) stand in for it (`W1[:, k-1] += W1[:, k]/2`, same for `k+1`). Crashes 49% ->
20% (healthy: 9%). Not a full fix: the neighbours see slightly different things.

## Lessons (consistent with the lander and walker experiments)
1. **Symmetries give exact edits** (left-hand traffic): perfect transfer with zero data.
2. **Networks don't learn the rule, they learn the cases they saw** (two speeds, not "drive at the limit"). Edits
   that ask for an unseen value fail; edits that map the new case onto a familiar one work.
3. **Editing perception works again**: "feel faster than you are" fixed the lander's heavy cargo, slowed the walker
   on ice, and now enforces school zones.
4. **Repairs of lost information are partial** (dead sensor): weights can reroute, not recreate, a signal.
5. The ambulance edit (a new behaviour: pull over) was not attempted; by the lessons above it would need training.

## Visuals (`visuals/`, `make_gifs.py`; each panel = whole city with route trail + close-up)
Routes are chosen by rules fixed in advance:
- `1_left_hand_traffic.gif`: first test route completed at home; home (right-hand) vs mirror-edited (left-hand).
- `2_school_zone.gif`: first test route with >= 10 s in a school zone; unedited vs school edit v2 (speed shown).
- `3_dead_sensor.gif`: first test route where the unpatched car crashes (chosen to show the failure); unpatched vs patched.

## Files
`city.py` simulator + expert, `render.py`, `train_bc.py`, `evaluate.py`, `edits.py`, `edits_eval.py`,
`nets/driver_final*.json` (final driver and edits), `nets/driver_v5_data.npz` (57 MB training data, deletable),
`v1_small_city/` (the earlier 3x3-city version).
