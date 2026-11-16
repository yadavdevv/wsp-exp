"""Top-down city driving: map, traffic lights, traffic, pedestrians, lidar, expert driver, observations.

Map: 4 x 4 grid of junctions 60 m apart, two-way roads (one lane each way, 3.5 m), side = +1 right-hand traffic
(lane on the right of travel), -1 left-hand traffic. Each junction runs 4 protected phases (one approach at a time,
5 s green incl. 1.5 s yellow + 3 s all-red clearance), so there are no turn conflicts.
Background cars ignore each other inside junction boxes (no gridlock among them) but always brake for the ego and pedestrians.

Every car (ego = index 0) uses a kinematic bicycle model. Others are driven by the expert. The ego can be driven by
the expert (data collection) or by a network policy.

Observation for a car (25 numbers, all in the car's own frame; "right" = + for the reference road):
   0 lateral offset from the road centreline / 3.5 (+ = right of travel)   1 heading error (rad)
   2 speed / 15                       3..29 lidar, 27 rays at ANGLES (dense in front), distance / 30 (1 = nothing)
  30..32 route command at next junction: left, straight, right (one-hot)
  33 distance to stop line / 50 (0 inside the junction)   34 light ahead says stop (red or yellow)
  35 speed limit / 15   36 inside a junction   37 junction ahead occupied ("don't block the box")
  38 school zone (0 in training)   39 ambulance behind (0 in training)
Reference road: the current road; inside a junction box, the exit road.
Action: steer, accel (- = brake), signal (- left, + right), each in [-1, 1].
"""
import numpy as np

GRID, BLOCK, HALF_ROAD, LANE, BOX = 4, 60.0, 3.5, 1.75, 8.0   # BOX: right-turn radius 6.25 > min 4.9
DT, WHEELBASE, MAX_STEER, V_MAX = 0.1, 2.7, 0.5, 15.0
DIRS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], float)        # N E S W
_HALF = [3, 6, 9, 12, 15, 19, 23, 28, 34, 42, 52, 68, 90]   # dense in front: a pedestrian can't hide between rays
ANGLES = np.radians([-a for a in _HALF[::-1]] + [0] + _HALF)
N_RAYS = len(ANGLES)
RANGE, CAR_R, CAR_OFF, PED_R = 30.0, 1.0, 1.2, 0.4
LIMITS = [8.3, 13.9]                                              # 30 and 50 km/h in m/s
SCHOOL_LIMIT = 5.56                                               # 20 km/h
N_OBS, N_ACT = 13 + N_RAYS, 3
YELLOW = 1.5   # last seconds of green shown as yellow (stop if you can)


def node_xy(n):
    return np.array([n[0] * BLOCK, n[1] * BLOCK])


def neighbours(n):
    return [(n[0] + int(d[0]), n[1] + int(d[1])) for d in DIRS
            if 0 <= n[0] + d[0] < GRID and 0 <= n[1] + d[1] < GRID]


def dir_index(a, b):
    d = node_xy(b) - node_xy(a)
    d /= np.linalg.norm(d)
    return int(np.argmax(DIRS @ d))


def right_of(d):
    return np.array([d[1], -d[0]])


def turn_type(a, b, c):
    d1, d2 = DIRS[dir_index(a, b)], DIRS[dir_index(b, c)]
    z = d1[0] * d2[1] - d1[1] * d2[0]
    return 0 if z > 0.5 else 2 if z < -0.5 else 1            # 0 left, 1 straight, 2 right


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Car:
    def __init__(self, a, b, c, u, v, side):
        self.a, self.b, self.c = a, b, c
        d = DIRS[dir_index(a, b)]
        self.pos = node_xy(a) + d * u + side * LANE * right_of(d)
        self.psi = float(np.arctan2(d[1], d[0]))
        self.v = v
        self.signal = 0.0
        self.stuck = 0
        self.make_path(side)

    def make_path(self, side):
        A, B, C = node_xy(self.a), node_xy(self.b), node_xy(self.c)
        d1, d2 = DIRS[dir_index(self.a, self.b)], DIRS[dir_index(self.b, self.c)]
        r1, r2 = side * LANE * right_of(d1), side * LANE * right_of(d2)
        p_in, p_out = B - BOX * d1 + r1, B + BOX * d2 + r2
        lane1 = A + BOX * d1 + r1 + np.outer(np.arange(0, BLOCK - 2 * BOX, 1.0), d1)
        if abs(d1 @ d2) > 0.5:
            ctrl = (p_in + p_out) / 2
        else:  # intersection of the two lane lines
            M = np.stack([d1, -d2], 1)
            s = np.linalg.solve(M, p_out - p_in)
            ctrl = p_in + s[0] * d1
        t = np.linspace(0, 1, 12)[:, None]
        arc = (1 - t) ** 2 * p_in + 2 * (1 - t) * t * ctrl + t ** 2 * p_out
        lane2 = p_out + np.outer(np.arange(1, 12, 1.0), d2)
        self.path = np.concatenate([lane1, arc, lane2])
        self.n_arc = len(lane1)                      # path index where the junction arc starts
        self.n_exit = len(lane1) + len(arc)          # path index where the exit road starts
        self.idx = 0
        self.turn = turn_type(self.a, self.b, self.c)

    def circles(self):
        h = np.array([np.cos(self.psi), np.sin(self.psi)])
        return np.stack([self.pos + CAR_OFF * h, self.pos - CAR_OFF * h])


class World:
    def __init__(self, seed=0, side=1, n_cars=14, n_peds=10, school=False, ambulance=False, route_len=8):
        self.rng = np.random.default_rng(seed)
        self.side, self.t, self.n_peds_target = side, 0.0, n_peds
        nodes = [(i, j) for i in range(GRID) for j in range(GRID)]
        self.phase_off = {n: self.rng.uniform(0, 32) for n in nodes}
        self.limit = {(a, b): self.rng.choice(LIMITS) for a in nodes for b in neighbours(a)}
        self.school = {(a, b): False for a in nodes for b in neighbours(a)}
        if school:   # three roads (both directions) become school zones: posted 30 km/h, required 20 km/h
            for a, b in [list(self.limit)[i] for i in self.rng.choice(len(self.limit), 3, replace=False)]:
                self.school[(a, b)] = self.school[(b, a)] = True
                self.limit[(a, b)] = self.limit[(b, a)] = LIMITS[0]
        self.cars = []
        for k in range(n_cars + 1):
            for _ in range(100):
                a = nodes[self.rng.integers(len(nodes))]
                b = neighbours(a)[self.rng.integers(len(neighbours(a)))]
                c = self.next_node(a, b)
                car = Car(a, b, c, self.rng.uniform(8, 40), self.rng.uniform(3, 7) if k else 5.0, side)
                if all(np.linalg.norm(car.pos - o.pos) > 10 for o in self.cars):
                    self.cars.append(car)
                    break
        self.route_left = route_len
        self.peds = []           # [pos, vel, life]
        self.events = {"collision": False, "offroad": False, "red_runs": 0, "done": False, "segments": 0}

    # ---- map helpers
    def next_node(self, a, b):
        opts = [c for c in neighbours(b) if c != a]
        return opts[self.rng.integers(len(opts))]

    def green(self, node, approach_dir, margin=0.0):
        ph = (self.t + self.phase_off[node]) % 32
        return int(ph // 8) == approach_dir and ph % 8 < 5 - margin

    def box_occupied(self, node, skip):
        return any(self.in_box(c.pos, node) for k, c in enumerate(self.cars) if k != skip)

    def on_road(self, pos, margin=0.3):
        """On any road surface (map geometry only)."""
        lim, top = HALF_ROAD + margin, (GRID - 1) * BLOCK
        near_v = np.min(np.abs(pos[0] - np.arange(GRID) * BLOCK)) < lim and -lim < pos[1] < top + lim
        near_h = np.min(np.abs(pos[1] - np.arange(GRID) * BLOCK)) < lim and -lim < pos[0] < top + lim
        return bool(near_v or near_h)

    def in_box(self, pos, node):
        p = node_xy(node)
        return abs(pos[0] - p[0]) < BOX and abs(pos[1] - p[1]) < BOX

    # ---- perception
    def obstacles(self, skip):
        cs, rs = [], []
        for k, o in enumerate(self.cars):
            if skip > 0 and k > 0 and self.in_box(o.pos, o.b):
                continue      # background cars don't see each other inside junctions
            if k != skip:
                cs.append(o.circles()); rs += [CAR_R, CAR_R]
        for p in self.peds:
            cs.append(p[0][None]); rs.append(PED_R)
        return (np.concatenate(cs), np.array(rs)) if cs else (np.zeros((0, 2)), np.zeros(0))

    def lidar(self, k):
        car = self.cars[k]
        C, R = self.obstacles(k)
        ang = car.psi + ANGLES
        D = np.stack([np.cos(ang), np.sin(ang)], 1)                   # (13, 2)
        rel = C - car.pos                                              # (M, 2)
        b = D @ rel.T                                                  # (13, M)
        c = (rel ** 2).sum(1) - R ** 2
        disc = b ** 2 - c
        t = np.where((disc >= 0) & (b > 0), b - np.sqrt(np.maximum(disc, 0)), np.inf)
        return np.minimum(t.min(1) if len(R) else np.full(N_RAYS, np.inf), RANGE)

    def reference(self, car):
        if car.idx >= car.n_arc:             # on the junction arc or beyond: the exit road is the reference
            a, b = car.b, car.c
        else:
            a, b = car.a, car.b
        d = DIRS[dir_index(a, b)]
        e = (car.pos - node_xy(a)) @ right_of(d)
        return e, wrap(car.psi - np.arctan2(d[1], d[0])), (a, b)

    def stop_dist(self, car):
        d = DIRS[dir_index(car.a, car.b)]
        if self.in_box(car.pos, car.b):
            return 0.0
        return max(0.0, (node_xy(car.b) - BOX * d - car.pos) @ d)

    def observe(self, k, lidar=None):
        car = self.cars[k]
        e, he, seg = self.reference(car)
        lid = self.lidar(k) if lidar is None else lidar
        cmd = np.zeros(3); cmd[car.turn] = 1
        red = 0.0 if self.green(car.b, dir_index(car.a, car.b), YELLOW) else 1.0
        inbox = float(self.in_box(car.pos, car.b))
        occ = float(self.box_occupied(car.b, k)) * (1 - inbox)
        return np.concatenate([[e / HALF_ROAD, he, car.v / 15], lid / RANGE, cmd,
                               [min(self.stop_dist(car) / 50, 1.0), red * (1 - inbox), self.limit[seg] / 15, inbox,
                                occ, float(self.school[seg]), 0.0]])

    # ---- expert driver (privileged path, but only lidar for obstacles)
    def expert(self, k, lidar=None):
        car = self.cars[k]
        lid = self.lidar(k) if lidar is None else lidar
        # steering: pure pursuit on the planned path
        i = int(np.argmin(np.linalg.norm(car.path - car.pos, axis=1)))
        Ld = max(4.0, 0.6 * car.v)
        j = min(i + int(Ld), len(car.path) - 1)
        tgt = car.path[j] - car.pos
        alpha = wrap(np.arctan2(tgt[1], tgt[0]) - car.psi)
        steer = np.clip(np.arctan2(2 * WHEELBASE * np.sin(alpha), Ld) / MAX_STEER, -1, 1)
        # speed: limit, turns, obstacles in the lane corridor, red light
        _, _, seg = self.reference(car)
        v_t = self.limit[seg]
        sd = self.stop_dist(car)
        turning = car.turn != 1
        if turning and (sd < 20 or self.in_box(car.pos, car.b)):
            v_t = min(v_t, 5.0)
        hit = lid < RANGE                          # obstacles near the path I'm about to drive (not just straight ahead)
        if hit.any():
            pts = car.pos + lid[hit, None] * np.stack([np.cos(car.psi + ANGLES[hit]), np.sin(car.psi + ANGLES[hit])], 1)
            P = car.path[i:i + 25]
            dist = np.linalg.norm(pts[:, None] - P[None], axis=2)          # (hits, path points)
            near = dist.min(1) < 1.8
            if near.any():
                along = dist[near].argmin(1).min()                          # path points are ~1 m apart
                v_t = min(v_t, np.sqrt(2 * 3.0 * max(0.0, along - 5.0)))
        stop = not self.green(car.b, dir_index(car.a, car.b), YELLOW) or (sd > 0 and self.box_occupied(car.b, k))
        if stop and sd > 0 and car.v ** 2 / 12 < sd + 0.5:
            v_t = min(v_t, np.sqrt(2 * 3.0 * max(0.0, sd - 1.0)))
        accel = np.clip(v_t - car.v, -1, 1)
        signal = 0.0
        if turning and (sd < 30 or self.in_box(car.pos, car.b)):
            signal = -1.0 if car.turn == 0 else 1.0
        return np.array([steer, accel, signal])

    # ---- dynamics
    def move(self, car, act):
        steer, accel = np.clip(act[0], -1, 1), np.clip(act[1], -1, 1)
        a = accel * (3.0 if accel > 0 else 6.0)
        car.v = float(np.clip(car.v + a * DT, 0, V_MAX))
        car.psi = float(wrap(car.psi + car.v / WHEELBASE * np.tan(steer * MAX_STEER) * DT))
        car.pos = car.pos + car.v * DT * np.array([np.cos(car.psi), np.sin(car.psi)])
        car.signal = float(act[2])
        car.stuck = car.stuck + 1 if car.v < 0.1 else 0
        i = int(np.argmin(np.linalg.norm(car.path - car.pos, axis=1)))
        car.idx = i
        if i >= car.n_exit and not self.in_box(car.pos, car.b):   # finished the junction: advance the route
            car.a, car.b = car.b, car.c
            car.c = self.next_node(car.a, car.b)
            car.make_path(self.side)
            return True
        return False

    def step(self, ego_action):
        ego = self.cars[0]
        sd_before = self.stop_dist(ego)
        red_before = not self.green(ego.b, dir_index(ego.a, ego.b))
        acts = [ego_action] + [self.expert(k) for k in range(1, len(self.cars))]
        for k, (car, act) in enumerate(zip(self.cars, acts)):
            advanced = self.move(car, act)
            if k == 0 and advanced:
                self.events["segments"] += 1
                self.route_left -= 1
        # pedestrians
        car_pts = np.concatenate([c.circles() for c in self.cars])
        for p in self.peds:
            nxt = p[0] + p[1] * DT
            gap = np.min(np.linalg.norm(car_pts - nxt, axis=1))
            on_kerb = not self.on_road(p[0], margin=0.0)
            if gap > (18.0 if on_kerb else 2.5):     # look before stepping off the kerb; never walk into a car
                p[0] = nxt
            p[2] -= DT
        self.peds = [p for p in self.peds if p[2] > 0]
        for _ in range(3):
            if len(self.peds) < self.n_peds_target:
                self.spawn_ped()
        self.t += DT
        # ego events
        if sd_before > 0 and self.in_box(ego.pos, ego.b) and red_before:
            self.events["red_runs"] += 1
        C, R = self.obstacles(0)
        if len(R) and (np.linalg.norm(ego.circles()[:, None] - C[None], axis=2) < CAR_R + R[None]).any():
            self.events["collision"] = True
        if not self.on_road(ego.pos):
            self.events["offroad"] = True
        if self.route_left <= 0:
            self.events["done"] = True
        # other cars that crash or leave get respawned elsewhere (rare)
        return self.events

    def spawn_ped(self):
        nodes = [(i, j) for i in range(GRID) for j in range(GRID)]
        a = nodes[self.rng.integers(len(nodes))]
        b = neighbours(a)[self.rng.integers(len(neighbours(a)))]
        d = DIRS[dir_index(a, b)]
        s = self.rng.choice([-1, 1])
        start = node_xy(a) + d * self.rng.uniform(15, 45) + s * 5.5 * right_of(d)
        if min(np.linalg.norm(start - c.pos) for c in self.cars) < 25:
            return
        self.peds.append([start, -s * 1.3 * right_of(d), 11 / 1.3 + 6])     # crossing time + 6 s patience
