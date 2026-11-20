"""CSCI 3302 Lab 7 - Stretch 3: explore, map, find, pick and place.   (STARTER CODE)

Mission
  1. Drive through the apartment (provided tour) and build a map with the 3-D lidar
     (16 layers x 360 deg), like your Lab 5 map. Anything the robot could hit (walls, chair
     legs, table tops, the bed) ends up in the map. EXTRA CREDIT: also build a 3-D map.
  2. Find the pose (x, y, yaw) of the red pick object in that map. For grading, the object is
     moved to a random spot with a random yaw, so do not hard-code its position.
  3. Plan a collision-free path to a grasp pose, line up the arm with the object and grasp it.
  4. Carry it to the goal stand and put it down on the green pad.

Everything without a TODO is PROVIDED: read it, you should not need to change it.
Every function with a TODO is yours. Each one stops the program with an AssertionError until you
write it, so the error message tells you what to do next. The first one you will see is
VoxelMap.occupied (the map display draws your map from the very first step). Work through the
parts in order:
  PART 1  VoxelMap.update, occupied, free                lidar map (Lab 5); 3-D map = extra credit
  PART 2  Planner.update, from_robot, path_to             planning map + Dijkstra / A*
  PART 3  obstacle_points, footprint_dist, follow,        path following (Lab 3 controller),
          rotate_to, go_to                                obstacle checks, re-planning
  PART 4  rect_fit, object_candidates, inspect_candidate  find the box and its pose in your map
  PART 5  base_pose_for, choose_base_pose, dock_at,       inverse kinematics, pick and place
          pick, place
PROVIDED: the mapping tour (explore) drives through every room with your PART 2 + 3 code while
your PART 1 map grows.
The map display (top left of the 3-D view) shows your map while the robot runs.

Coordinate frames
  world : Webots ENU frame (x east, y north, z up). GPS gives the base origin, the IMU the yaw.
  base  : origin on the floor between the two drive wheels, x forward, y left, z up.
          The telescoping arm extends along -y (to the robot's RIGHT side).
"""
import math
import os

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from controller import Robot, Display

# =====================================================================================
# PROVIDED: robot constants (measured from the Stretch 3 model)
# =====================================================================================
WHEEL_RADIUS = 0.0508          # [m]
WHEEL_BASE = 0.3407            # [m] distance between the wheels
MAX_WHEEL_SPEED = 6.0          # [rad/s]  -> 0.30 m/s
LIDAR_POS = (0.004, 0.0, 0.1924)   # lidar position in the base frame [m]
ROBOT_HEIGHT = 1.5             # [m] anything lower than this can hit the robot
# Robot outline in the base frame (base + stowed wrist/gripper): xmin xmax ymin ymax.
ROBOT_FOOTPRINT = (-0.276, 0.10, -0.22, 0.171)
# Lidar points inside this box hit the robot itself (mast, stowed gripper): ignore them.
SELF_BOX = (-0.31, 0.13, -0.26, 0.20)
# Grasp center (between the fingertips) in the base frame for lift = 0, arm = 0, wrist = 0.
# It moves straight up with the lift and straight along -y with the arm:
#   grasp = (GRASP_X, GRASP_Y0 - arm, GRASP_Z0 + lift)
GRASP_X, GRASP_Y0, GRASP_Z0 = -0.0213, -0.4148, 0.1145
LIFT_RANGE = (0.0, 1.1)
ARM_RANGE = (0.0, 0.52)

# =====================================================================================
# PROVIDED: task constants
# =====================================================================================
OBJECT_SIZE = (0.06, 0.14, 0.25)        # pick object: width, length, height [m] (it stands on the floor)
GOAL_PAD = (-9.55, -7.50, 0.444)        # center of the green pad on the goal stand (top z)
MAP_BOUNDS = (-12.8, 0.8, -13.6, 0.6)   # xmin xmax ymin ymax of the apartment [m]

# =====================================================================================
# Parameters: used by the provided code, except FLOOR_Z and R_TIGHT (handy for your own code)
# =====================================================================================
VOX = 0.05                     # voxel / grid cell size [m]
MAP_TOP = 1.6                  # the 3-D map covers z = 0 .. 1.6 m
FLOOR_Z = 0.06                 # [m] lidar points lower than this are floor
OBST_HITS = 3                  # voxels() draws voxels with at least this many hits (map3d.png)
R_LETHAL = 0.31                # [m] default robot radius for planning (plan())
R_TIGHT = 0.27                 # [m] smaller robot radius (tight spots)
R_SWING = 0.33                 # [m] rear corners sweep this radius when the base turns in place
V_MAX = 0.26                   # [m/s] top driving speed
LIFT_TRAVEL = 0.22             # lift height while driving (gripper stowed, see stow())
WRIST_STOW = math.pi - 0.3     # wrist yaw that tucks the gripper above the base
# Add your own parameters (thresholds, gains, speeds, ...) here.

viz = dict(path=None, targets=None, obj=None, phase='starting', last=-1.0)


def log(*a):
    """Print a message with the simulated time in front, e.g. log('found', x) -> "[  12.3s] found 1.5"."""
    print(f'[{robot.getTime():7.1f}s]', *a, flush=True)


def wrap(a):
    """Wrap an angle to [-pi, pi). Use it for heading errors: wrap(theta_goal - theta)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# =====================================================================================
# PROVIDED: robot interface
# =====================================================================================
robot = Robot()
TIMESTEP = 32                  # controller step [ms] (world basicTimeStep is 16 ms)
LIDAR_PERIOD = 96              # lidar update [ms] (~10 Hz)


class Stretch:
    def __init__(self):
        """Find and enable all devices (wheels, lift, arm, wrist, gripper, sensors, lidar, map display).
        You never call this yourself: use the global object `stretch`."""
        g = robot.getDevice
        self.wheels = [g('joint_left_wheel'), g('joint_right_wheel')]
        for w in self.wheels:
            w.setPosition(float('inf'))
            w.setVelocity(0.0)
        self.lift_m = g('joint_lift')
        self.arm_m = [g(f'joint_arm_l{i}') for i in range(4)]
        self.yaw_m, self.pitch_m, self.roll_m = g('joint_wrist_yaw'), g('joint_wrist_pitch'), g('joint_wrist_roll')
        self.fingers = [g('joint_gripper_finger_left'), g('joint_gripper_finger_right')]
        self.sensors = {}
        for name in ['joint_lift', 'joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3',
                     'joint_wrist_yaw', 'joint_gripper_finger_left', 'joint_gripper_finger_right']:
            s = g(name + '_sensor')
            s.enable(TIMESTEP)
            self.sensors[name] = s
        self.gps = g('gps'); self.gps.enable(TIMESTEP)
        self.imu = g('imu'); self.imu.enable(TIMESTEP)
        self.lidar = g('lidar'); self.lidar.enable(LIDAR_PERIOD)
        self.display = g('map')
        # ray directions of the 3-D lidar: layer k (0 = top) and ray i in that layer
        n, nl = self.lidar.getHorizontalResolution(), self.lidar.getNumberOfLayers()
        vfov = self.lidar.getVerticalFov()
        az = math.pi - np.arange(n) * (2 * math.pi / n)                  # counter-clockwise
        el = vfov / 2 - np.arange(nl) * (vfov / (nl - 1))                  # elevation, top layer first
        self.ray_dir = np.stack([np.cos(el)[:, None] * np.cos(az)[None, :],
                                 np.cos(el)[:, None] * np.sin(az)[None, :],
                                 np.sin(el)[:, None] * np.ones(n)[None, :]], axis=-1).reshape(-1, 3)
        self._last_ranges = None

    # --- sensing ---
    def pose(self):
        """Robot pose for planning and driving.

        Returns (x, y, yaw): base origin in the world frame [m] and heading [rad, counter-clockwise
        from the world x axis]. This is ground truth (GPS + IMU), as in Lab 5."""
        x, y, _ = self.gps.getValues()
        return x, y, self.imu.getRollPitchYaw()[2]

    def pose3d(self):
        """Full 3-D pose of the base: (position, R) with position = array (3,) [m] and R = 3 x 3
        rotation matrix (base frame -> world frame).

        Use it (not pose()) to turn lidar points into world points: to_world(pts, stretch.pose3d()).
        The base pitches a little when it brakes; far lidar points are only right with roll and
        pitch included."""
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                                  math.cos(yaw), math.sin(yaw))
        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])
        return np.array(self.gps.getValues()), R

    def scan_points(self):
        """The newest 3-D lidar scan as an N x 3 array of points (x, y, z) in the BASE frame [m].

        Returns None if the lidar has no new scan this step (it scans about 10 times a second), so
        a scan is only used once, together with the pose of the same step. Rays without a return
        are dropped, and so are "mixed" rays on depth edges: a ray that grazes an edge returns a
        range halfway between the near object and the background, which would put a fake
        obstacle in free space. (Floor points are kept.) The robot's own body is still in the
        scan: drop points inside SELF_BOX yourself.

        You normally do not call this directly: sense() calls it every step and keeps the newest
        scan in the global `last_pts`."""
        r = np.array(self.lidar.getRangeImage(), dtype=float)
        if self._last_ranges is not None and np.array_equal(r, self._last_ranges):
            return None
        self._last_ranges = r
        img = r.reshape(self.lidar.getNumberOfLayers(), -1)
        gap = np.maximum(0.1, 0.05 * img)

        def between(a, b):
            with np.errstate(invalid='ignore'):          # inf - inf where rays have no return
                return (img > np.minimum(a, b) + gap) & (img < np.maximum(a, b) - gap)
        side = between(np.roll(img, 1, axis=1), np.roll(img, -1, axis=1))
        inf_row = np.full((1, img.shape[1]), np.inf)
        updown = between(np.vstack([inf_row, img[:-1]]), np.vstack([img[1:], inf_row]))
        pts = np.asarray(LIDAR_POS) + self.ray_dir * np.where(np.isfinite(r), r, 0.0)[:, None]
        on_floor = pts[:, 2] < 0.08
        ok = np.isfinite(r) & ~side.reshape(-1) & ~(updown.reshape(-1) & ~on_floor)
        return pts[ok]

    # --- base ---
    def drive(self, v, w):
        """Drive the base: v = forward speed [m/s], w = turn rate [rad/s, counter-clockwise].

        The wheel speeds are scaled down together if one would exceed MAX_WHEEL_SPEED (top speed
        ~0.30 m/s). The command stays active until you send another one: call stretch.stop()
        (= drive(0, 0)) to stop. Example: stretch.drive(0.2, 0.0) drives straight ahead."""
        wl = (v - w * WHEEL_BASE / 2) / WHEEL_RADIUS
        wr = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS
        k = max(1.0, abs(wl) / MAX_WHEEL_SPEED, abs(wr) / MAX_WHEEL_SPEED)
        self.wheels[0].setVelocity(wl / k)
        self.wheels[1].setVelocity(wr / k)

    def stop(self):
        """Stop the base (same as drive(0, 0))."""
        self.drive(0.0, 0.0)

    # --- arm ---
    def set_lift(self, z):
        """Move the lift (arm height) to z [m], clipped to LIFT_RANGE (0 .. 1.1 m). Returns at once;
        use move_arm(lift=z) to wait until it is there."""
        self.lift_m.setPosition(float(np.clip(z, *LIFT_RANGE)))

    def lift(self):
        """Current lift position [m]."""
        return self.sensors['joint_lift'].getValue()

    def set_arm(self, e):
        """Extend the telescoping arm to e [m] (0 .. 0.52 m), out to the robot's RIGHT side (-y).
        Returns at once; use move_arm(arm=e) to wait until it is there."""
        e = float(np.clip(e, *ARM_RANGE))
        for m in self.arm_m:
            m.setPosition(e / 4)

    def arm(self):
        """Current arm extension [m] (sum of the four telescope joints)."""
        return sum(self.sensors[f'joint_arm_l{i}'].getValue() for i in range(4))

    def set_wrist(self, yaw, pitch=0.0, roll=0.0):
        """Set the wrist angles [rad]. yaw = 0 points the gripper straight along the arm
        (down/out, ready to grasp); WRIST_STOW tucks it above the base for driving. Returns at once;
        use move_arm(yaw=...) to wait."""
        self.yaw_m.setPosition(yaw)
        self.pitch_m.setPosition(pitch)
        self.roll_m.setPosition(roll)

    def wrist_yaw(self):
        """Current wrist yaw [rad]."""
        return self.sensors['joint_wrist_yaw'].getValue()

    def set_gripper(self, a):
        """Set both finger angles [rad]: 0.45 = wide open (~18 cm), 0 = closed, -0.25 = squeeze.
        Prefer open_gripper() / close_gripper(), which also wait."""
        for f in self.fingers:
            f.setPosition(a)

    def fingers_pos(self):
        """Current (left, right) finger angles [rad]."""
        return (self.sensors['joint_gripper_finger_left'].getValue(),
                self.sensors['joint_gripper_finger_right'].getValue())


stretch = Stretch()


def to_world(pts, pose3d):
    """Turn base-frame points into world-frame points.

    pts    : N x 3 array of points in the base frame (e.g. a lidar scan)
    pose3d : (position, R) from stretch.pose3d() of the SAME step as the scan
    Returns an N x 3 array in the world frame. Example:
        w = to_world(pts, stretch.pose3d());  x, y, z = w[:, 0], w[:, 1], w[:, 2]"""
    t, R = pose3d
    return pts @ R.T + t


def step():
    """Advance the simulation by one control step (TIMESTEP = 32 ms). Exits cleanly when Webots stops.
    In your own loops, call step() and then sense() every iteration."""
    if robot.step(TIMESTEP) == -1:
        raise SystemExit(0)


def wait(seconds, until=None):
    """Let the simulation run for `seconds` (the map display keeps updating).

    until : optional function with no arguments; stop early as soon as it returns True.
    Returns True if `until` became True, False if the time ran out.
    Example: wait(2.0, lambda: stretch.lift() > 0.3)"""
    t_end = robot.getTime() + seconds
    while robot.getTime() < t_end:
        step()
        viz_tick()
        if until is not None and until():
            return True
    return False


def open_gripper():
    """Open the gripper wide (~18 cm between the fingertips) and wait until it is open."""
    stretch.set_gripper(0.45)
    wait(2.0, lambda: min(stretch.fingers_pos()) > 0.43)


def close_gripper():
    """Close the gripper and wait until the fingers stop.

    Returns True if something is held (the fingers stopped on an object), False if they closed
    on each other (missed). The fingers then hold slightly inside the contact angle so the object
    does not slip."""
    stretch.set_gripper(-0.25)
    last = stretch.fingers_pos()
    for _ in range(int(3.0 / (TIMESTEP / 1000))):
        step()
        cur = stretch.fingers_pos()
        if max(abs(c - l) for c, l in zip(cur, last)) < 1e-4:
            break
        last = cur
    cur = stretch.fingers_pos()
    for f, a in zip(stretch.fingers, cur):
        f.setPosition(a - 0.03)
    wait(0.5)
    return min(cur) > -0.18          # fingers stopped on the object, not on each other


def move_arm(lift=None, arm=None, yaw=None, timeout=8.0):
    """Move lift / arm / wrist yaw (any you give, others stay) and WAIT until they get there
    (or `timeout` seconds pass).

    lift [m] (0 .. 1.1), arm [m] (0 .. 0.52), yaw [rad]. Example:
        move_arm(lift=0.3, arm=0.2)   # both at once, returns when both are there"""
    if lift is not None:
        stretch.set_lift(lift)
    if arm is not None:
        stretch.set_arm(arm)
    if yaw is not None:
        stretch.set_wrist(yaw)

    def done():
        ok = True
        if lift is not None:
            ok &= abs(stretch.lift() - lift) < 0.005
        if arm is not None:
            ok &= abs(stretch.arm() - arm) < 0.005
        if yaw is not None:
            ok &= abs(stretch.wrist_yaw() - yaw) < 0.02
        return ok
    wait(timeout, done)


# =====================================================================================
# PART 1: mapping with the lidar (2-D map; 3-D map = extra credit)
# =====================================================================================
class VoxelMap:
    """Lidar map on a VOX grid.
    Required (2-D):    self.col_hits  = lidar hits per cell (any height up to the robot's top),
                       self.free_count = how often each floor cell was seen empty.
    Extra credit (3-D): self.hits = lidar hits per voxel (row = y, column = x, layer = z).
                       If it is filled, you also get map3d.npy / map3d.png, obstacle heights on
                       the map display, and a faster box search (detect_object)."""

    def __init__(self, bounds):
        """Empty map over `bounds` = (xmin, xmax, ymin, ymax) [m] with VOX cells; nx, ny, nz cells.
        You use the global `vmap`."""
        self.x0, x1, self.y0, y1 = bounds
        self.nx = int(round((x1 - self.x0) / VOX))
        self.ny = int(round((y1 - self.y0) / VOX))
        self.nz = int(round(MAP_TOP / VOX))
        self.col_hits = np.zeros((self.ny, self.nx), np.int32)          # 2-D: hits per cell
        self.free_count = np.zeros((self.ny, self.nx), np.int32)        # 2-D: seen empty
        self.hits = np.zeros((self.ny, self.nx, self.nz), np.uint16)   # 3-D (extra credit)

    def cell(self, x, y):
        """World (x, y) [m] -> grid cell (row iy, column ix). Rows go with y, columns with x:
        array[iy, ix]. Check vmap.inside(iy, ix) before you index with it."""
        return int((y - self.y0) / VOX), int((x - self.x0) / VOX)

    def inside(self, iy, ix):
        """True if cell (iy, ix) is inside the map."""
        return 0 <= iy < self.ny and 0 <= ix < self.nx

    def center(self, iy, ix):
        """Grid cell (row iy, column ix) -> world (x, y) [m] of the cell center."""
        return self.x0 + (ix + 0.5) * VOX, self.y0 + (iy + 0.5) * VOX

    def _index(self, x, y):
        """Vectorized cell(): arrays x, y [m] -> arrays (iy, ix, ok), ok = inside the map.
        Use it to update many points at once: iy, ix, ok = self._index(x, y);
        np.add.at(self.col_hits, (iy[ok], ix[ok]), 1)"""
        ix = np.floor((x - self.x0) / VOX).astype(np.int64)
        iy = np.floor((y - self.y0) / VOX).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy, ix, ok

    def update(self, pose3d, pts):
        """Add one lidar scan to the map.
        pts = lidar points in the base frame (N x 3), pose3d = stretch.pose3d() of the same step.

        TODO: like the Lab 5 map, every lidar hit is evidence for an obstacle. Update
        self.col_hits (hits per cell, for obstacles between the floor and the top of the robot)
        and self.free_count (how often each floor cell was seen empty).
        Watch out for points that hit the robot itself (SELF_BOX).
        EXTRA CREDIT: also count the hits per voxel in self.hits (3-D map).
        """

        lidarPtsWorld = to_world(pts, pose3d)
        (ix, iy, ok) = self._index(lidarPtsWorld[:, 0], lidarPtsWorld[:, 1]) #row is iy, column is ix
        for i in range(len(pts)): 
            for j in range(len(pts)): 
                if ok[i, j]: 
                    self.col_hits[iy[i], ix[j]] += 1

        assert False, 'TODO (PART 1): VoxelMap.update is not implemented yet'

    def occupied(self):
        """TODO: 2-D obstacle map (ny x nx, bool) from self.col_hits. Like the Lab 5 threshold:
        a few stray hits must not become obstacles."""
        assert False, 'TODO (PART 1): VoxelMap.occupied is not implemented yet'

    def top(self):
        """Height [m] of the highest 3-D map voxel (with >= 2 hits) in each cell (ny x nx), or None
        if there is no 3-D map (self.hits is empty; it is extra credit)."""
        if not self.hits.any():
            return None
        strong = self.hits >= 2
        return np.where(strong.any(axis=2), (self.nz - np.argmax(strong[:, :, ::-1], axis=2)) * VOX, 0.0)

    def free(self):
        """TODO: 2-D free-space map (ny x nx, bool). Cells never seen are unknown, not free."""
        assert False, 'TODO (PART 1): VoxelMap.free is not implemented yet'

    def voxels(self, min_hits=OBST_HITS):
        """Occupied voxels as an N x 3 array of (row, column, layer), only in occupied cells
        (used for map3d.png). Empty without a 3-D map."""
        return np.argwhere((self.hits >= min_hits) & self.occupied()[:, :, None])


vmap = VoxelMap(MAP_BOUNDS)
mapping = True                 # switched off while the arm is out (it would see itself)
last_pts = None                # newest lidar scan (base frame)


def viz_tick():
    """Redraw the map display about twice a second (called by sense() and wait())."""
    t = robot.getTime()
    if t - viz['last'] >= 0.5:
        viz['last'] = t
        draw_map()


def sense():
    """Read the lidar and update the map. Call it once every control step (after step()).

    If there is a new scan: keeps it in the global `last_pts` (base frame) and, unless mapping is
    switched off (global `mapping` = False, e.g. while the arm is out), adds it to the map with
    vmap.update(stretch.pose3d(), pts). Returns the new scan, or None."""
    global last_pts
    pts = stretch.scan_points()
    if pts is not None:
        last_pts = pts
        if mapping:
            vmap.update(stretch.pose3d(), pts)
    viz_tick()
    return pts


# =====================================================================================
# PART 2: planning map (inflation) + Dijkstra path planning
# =====================================================================================
class Planner:
    def __init__(self, vmap):
        """Planner on the grid of vmap. You use the global `planner`; see Planner.update and
        Planner.from_robot for what it must provide."""
        self.m = vmap
        self.ny, self.nx = vmap.ny, vmap.nx
        self.dist = None

    def update(self, r_lethal=R_LETHAL):
        """Build the planning map from the current map (vmap.occupied() and vmap.free()).
        TODO: compute and set
          self.passable  cells the robot's center may visit with obstacles r_lethal away
                         (ny x nx, bool). The provided code reads it (closest_reachable).
        The rest is yours to design. A planner usually also keeps, for example:
          self.occ, self.free  2-D occupied / free maps (ny x nx, bool)
          self.clear     distance [m] from every cell to the nearest obstacle or unknown cell
          self.cost      cost of every cell (higher close to obstacles)
        """
        assert False, 'TODO (PART 2): Planner.update is not implemented yet'

    def from_robot(self, x, y):
        """Shortest paths from the robot at world (x, y) to every cell (Dijkstra or A*).
        TODO: set self.dist (ny x nx path costs, np.inf = unreachable), keep whatever path_to()
        needs, and return self.dist.
        """
        assert False, 'TODO (PART 2): Planner.from_robot is not implemented yet'

    def path_to(self, iy, ix):
        """TODO: path from the robot to cell (row iy, column ix) as a list of world (x, y)
        points, or None if that cell cannot be reached."""
        assert False, 'TODO (PART 2): Planner.path_to is not implemented yet'


planner = Planner(vmap)


def plan(r_lethal=R_LETHAL):
    """Update the planning map from the current map and run your shortest-path search from the robot:
    planner.update(r_lethal) then planner.from_robot(x, y). After it, planner.passable and
    planner.dist (and whatever else your Planner.update sets) are up to date and
    planner.path_to(iy, ix) gives paths."""
    planner.update(r_lethal)
    rx, ry, _ = stretch.pose()
    planner.from_robot(rx, ry)


# =====================================================================================
# PART 3: path following (pure pursuit) with 3-D lidar obstacle checks
# =====================================================================================
def obstacle_points(pts):
    """TODO: x, y (base frame) of the lidar points (N x 3, base frame) that the robot could
    hit while driving. Return two arrays (px, py)."""
    assert False, 'TODO (PART 3): obstacle_points is not implemented yet'


def footprint_dist(px, py):
    """TODO: distance of each point (base frame) to the robot's outline (ROBOT_FOOTPRINT);
    0 for points inside it. Return an array."""
    assert False, 'TODO (PART 3): footprint_dist is not implemented yet'


def robust_min(d, k=3):
    """k-th smallest value of array d (9.0 if d has fewer than k values).

    Use it instead of d.min() for lidar distances: a real obstacle returns many rays, a single
    stray ray returns one, so robust_min ignores the k-1 closest outliers."""
    return float(np.partition(d, k - 1)[k - 1]) if len(d) >= k else 9.0


def gaps_front_rear(px, py):
    """Free distance [m] in front of and behind the robot: (front, rear).

    px, py : obstacle points in the base frame (from obstacle_points()).
    Uses footprint_dist() and robust_min(). Used by back_off(); handy in follow() too."""
    d = footprint_dist(px, py)
    return robust_min(d[px > ROBOT_FOOTPRINT[1] - 0.02]), robust_min(d[px < ROBOT_FOOTPRINT[0] + 0.02])


def mark_obstacle(px, py):
    """Write obstacle points (base frame, px, py arrays) into the map for sure, so the next plan
    avoids them. Call it when follow() stops because something is too close.

    Adds 1000 hits to vmap.col_hits in those cells (and to the 3-D map vmap.hits, but only if you
    built one: without the extra credit, vmap.hits stays empty). Points outside the map are skipped."""
    w = to_world(np.column_stack([px, py, np.zeros_like(px)]), stretch.pose3d())
    has_3d = bool(vmap.hits.any())           # is there a 3-D map (extra credit)?
    for qx, qy in w[:, :2]:
        iy, ix = vmap.cell(qx, qy)
        if not vmap.inside(iy, ix):
            continue
        vmap.col_hits[iy, ix] += 1000
        if has_3d:
            vmap.hits[iy, ix, 2] = min(60000, int(vmap.hits[iy, ix, 2]) + 1000)


def follow(path, v_max=V_MAX, tol=0.10, time_limit=None):
    """Drive along a list of world (x, y) points.
    TODO: steer toward a point a bit ahead on the path (your Lab 3 controller works). Return
    'reached' (within tol of the last point), 'blocked' (an obstacle is too close: stop before
    touching it) or 'timeout'. Call step() and sense() every iteration.
    """
    assert False, 'TODO (PART 3): follow is not implemented yet'


def fresh_scan():
    """Wait (up to ~10 steps) for the next lidar scan and return it (base frame, N x 3).
    Use it when the robot stands still and you need a scan from where it is now."""
    for _ in range(10):
        step()
        pts = sense()
        if pts is not None:
            return pts
    return last_pts


def back_off(dist=0.12):
    """Recovery after a stop: drive straight back (or forward) by up to `dist` [m], toward the side
    with more room, stopping early if that side gets close too. Call it after follow() returned
    'blocked', before you plan again."""
    gf, gr = gaps_front_rear(*obstacle_points(fresh_scan()))
    if gr > 0.10 and gr >= gf:
        v = -0.06
    elif gf > 0.10:
        v = 0.06
    else:
        log('recovery: boxed in, not moving')
        return
    x0, y0, _ = stretch.pose()
    t0 = robot.getTime()
    while math.hypot(stretch.pose()[0] - x0, stretch.pose()[1] - y0) < dist and robot.getTime() - t0 < 4:
        stretch.drive(v, 0.0)
        step()
        if sense() is not None:
            gf, gr = gaps_front_rear(*obstacle_points(last_pts))
            if (gr if v < 0 else gf) < 0.05:
                break
    stretch.stop()


def rotate_to(theta, tol=0.01, time_limit=12.0):
    """TODO: turn in place to the world heading theta [rad]."""
    assert False, 'TODO (PART 3): rotate_to is not implemented yet'


def go_to(x, y, theta=None, tol=0.10, attempts=6, time_limit=90):
    """TODO: plan with the current map and drive to world (x, y), then turn to theta if given.
    Handle a blocked path (the map changes as the robot drives). Return True if the robot got
    there, False otherwise. time_limit [s] is for each attempt."""
    assert False, 'TODO (PART 3): go_to is not implemented yet'


# =====================================================================================
# PROVIDED: mapping tour (the robot drives through the apartment while it maps, like Lab 5)
# =====================================================================================
# Waypoints through every room (x, y, look): with look=True the robot turns around once there,
# because the base hides the floor right behind the lidar. Driving between them uses YOUR
# planner (PART 2) and go_to / path follower (PART 3) on the map YOU are building (PART 1).
TOUR = [(-8.28, -4.92, True), (-8.93, -3.12, False), (-10.58, -2.67, True), (-8.68, -2.87, False),
        (-8.38, -1.02, True), (-8.62, -2.92, False), (-8.12, -4.77, False), (-6.43, -5.52, False),
        (-5.18, -4.72, False), (-4.93, -3.88, True), (-5.23, -4.77, False),
        (-5.33, -6.77, False), (-3.78, -7.52, True), (-1.78, -7.62, False), (-0.97, -7.22, True),
        (-2.12, -8.38, False), (-1.72, -10.42, True), (-1.82, -8.97, False), (-2.78, -7.62, False),
        (-4.58, -9.32, False), (-4.33, -10.52, True), (-4.93, -8.72, False), (-6.28, -9.18, True),
        (-7.03, -10.82, False), (-9.33, -11.67, True), (-6.78, -9.62, False), (-6.18, -7.82, False),
        (-7.43, -7.42, True), (-8.57, -7.32, True)]


def look_around():
    """Turn once in place (in three steps) if there is room to turn. The base hides the floor right
    behind the lidar, so turning fills in the map. Returns False (and does nothing) if something
    is inside the R_SWING circle."""
    px, py = obstacle_points(fresh_scan())
    if robust_min(np.hypot(px, py)[footprint_dist(px, py) > 0.0]) < R_SWING + 0.03:
        return False
    th = stretch.pose()[2]
    for k in (1, 2, 3):
        rotate_to(wrap(th + k * 2 * math.pi / 3), tol=0.05)
    return True


def closest_reachable(x, y, radius=1.5):
    """Reachable cell closest to world (x, y), within `radius` [m].

    Uses planner.dist (finite = reachable) and planner.passable, so call plan() first.
    Returns the world (x, y) of that cell, or None if there is none."""
    iy, ix = vmap.cell(x, y)
    r = int(radius / VOX)
    y0, x0 = max(0, iy - r), max(0, ix - r)
    sub = np.isfinite(planner.dist[y0:iy + r + 1, x0:ix + r + 1]) & planner.passable[y0:iy + r + 1, x0:ix + r + 1]
    if not sub.any():
        return None
    yy, xx = np.nonzero(sub)
    k = np.argmin((yy + y0 - iy) ** 2 + (xx + x0 - ix) ** 2)
    return vmap.center(yy[k] + y0, xx[k] + x0)


def explore():
    """The mapping tour: drive to every TOUR waypoint with YOUR go_to() while sense() keeps adding
    scans to YOUR map, and look_around() in each room. A waypoint that is not in known free space
    yet is approached as close as the map allows; one that cannot be reached within about two
    minutes is skipped. Prints progress ("tour waypoint k/N done")."""
    log('mapping tour...')
    viz['phase'] = 'mapping tour'
    t_start = robot.getTime()
    for k, (x, y, look) in enumerate(TOUR):
        viz['targets'] = [(0.0, 1, *vmap.cell(x, y))]
        for _ in range(2):
            plan()
            goal = (x, y) if planner.path_to(*vmap.cell(x, y)) is not None else closest_reachable(x, y)
            if goal is None:
                break
            if go_to(*goal, tol=0.25, attempts=2, time_limit=30) and math.hypot(goal[0] - x, goal[1] - y) < 0.3:
                break
            look_around()                         # see more, then try again
        if look:
            look_around()
        log(f'tour waypoint {k + 1}/{len(TOUR)} done')
    stretch.stop()
    viz['targets'] = None
    log(f'mapping tour took {(robot.getTime() - t_start) / 60:.1f} min')


# =====================================================================================
# PROVIDED: map check (runs after the mapping tour; your map must pass before the robot goes on)
# =====================================================================================
# Places that are obstacles in the apartment: walls, furniture, table tops / seats (overhangs:
# the floor below them is free, so only a map that uses points at all heights has them), and the
# goal stand. Each must be occupied in your map (within 10 cm).
MAP_CHECK_OBSTACLES = [(-1.68, -3.77), (-0.47, -12.27), (-9.68, -7.52), (-3.38, -6.42), (-5.88, -2.92),
                       (-11.32, -3.32), (-4.98, -11.27), (-7.98, -6.42), (-9.73, -9.57), (-7.58, -13.02),
                       (-4.68, -12.92), (-9.58, -6.12), (-3.53, -9.72), (-4.93, -0.52), (-2.28, -6.72),
                       (-4.47, -6.48), (-9.55, -7.50)]
MAP_CHECK_MIN_FREE = 65.0      # [m^2] free floor your map must cover (the apartment has ~83 m^2)
MAP_CHECK_MAX_SPECKS = 25      # lone obstacle specks (1-2 cells, nothing else within 0.3 m) allowed


def check_map():
    """Check the map after the mapping tour and stop with a report if it is not good enough.

    It checks five things (and prints PASS / FAIL for each):
      1. coverage   : at least MAP_CHECK_MIN_FREE m^2 of floor is free (vmap.free()).
      2. obstacles  : every point in MAP_CHECK_OBSTACLES is occupied (vmap.occupied()), at most one
                      may be missed. Table tops and seats only show up if you use lidar points at
                      all heights up to the top of the robot, not just a 2-D slice.
      3. free path  : no tour waypoint has an obstacle within 15 cm (the robot drove there, so
                      an obstacle there is a fake one).
      4. noise      : at most MAP_CHECK_MAX_SPECKS lone specks (noise that made it through your
                      threshold).
      5. planning   : with your planner (plan(), planner.dist), every room the tour stopped in can
                      be reached from where the robot is now.
    Returns True if all pass; raises AssertionError (with the report) if not.
    """
    occ, free = vmap.occupied(), vmap.free()
    report, ok = [], True

    def near(x, y, mask, r):
        iy, ix = vmap.cell(x, y)
        k = int(round(r / VOX))
        return bool(mask[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1].any())

    def line(name, passed, msg):
        nonlocal ok
        ok &= passed
        report.append(f'  [{"PASS" if passed else "FAIL"}] {name:<10} {msg}')

    area = free.sum() * VOX * VOX
    line('coverage', area >= MAP_CHECK_MIN_FREE, f'{area:.1f} m^2 free floor (need >= {MAP_CHECK_MIN_FREE:.0f})')
    missed = [p for p in MAP_CHECK_OBSTACLES if not near(p[0], p[1], occ, 0.10)]
    line('obstacles', len(missed) <= 1, f'{len(MAP_CHECK_OBSTACLES) - len(missed)}/{len(MAP_CHECK_OBSTACLES)} known '
         f'obstacles in the map' + (f', missing near {[(round(x, 2), round(y, 2)) for x, y in missed]}' if missed else ''))
    fake = [(x, y) for x, y, _ in TOUR if near(x, y, occ, 0.15)]
    line('free path', not fake, f'{len(fake)} tour waypoints with a (fake) obstacle within 15 cm'
         + (f': {fake}' if fake else ''))
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    sizes = ndimage.sum(occ, lab, index=np.arange(1, n + 1)) if n else np.zeros(0)
    specks = 0
    for k in np.nonzero(sizes <= 2)[0] + 1:
        comp = lab == k
        if not (ndimage.binary_dilation(comp, iterations=int(0.3 / VOX)) & occ & ~comp).any():
            specks += 1
    line('noise', specks <= MAP_CHECK_MAX_SPECKS, f'{specks} lone specks (allowed {MAP_CHECK_MAX_SPECKS})')
    plan()
    rooms = [(x, y) for x, y, look in TOUR if look]
    unreachable = []
    for x, y in rooms:
        iy, ix = vmap.cell(x, y)
        k = int(0.3 / VOX)
        if not np.isfinite(planner.dist[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1]).any():
            unreachable.append((x, y))
    line('planning', not unreachable, f'{len(rooms) - len(unreachable)}/{len(rooms)} tour rooms reachable with '
         f'your planner' + (f', not: {unreachable}' if unreachable else ''))
    log('MAP CHECK ' + ('PASSED' if ok else 'FAILED') + '\n' + '\n'.join(report))
    assert ok, 'MAP CHECK FAILED: fix your map (PART 1) / planner (PART 2), see the report above'
    return True


# =====================================================================================
# PART 4: find the pick object in the map (pose = x, y, yaw of its long axis)
# =====================================================================================
def rect_fit(pts):
    """TODO: minimum-area rectangle around 2-D points (N x 2).
    Return (cx, cy, angle_of_long_axis in [0, pi), length, width)."""
    assert False, 'TODO (PART 4): rect_fit is not implemented yet'


def object_candidates():
    """TODO: world (x, y) of the places in the map that could be the pick object, most likely
    first. (vmap.top() gives obstacle heights if you built the 3-D map, else None.)"""
    assert False, 'TODO (PART 4): object_candidates is not implemented yet'


def object_points(pts, guess, radius=0.16):
    """Lidar points near a guessed object position, in the WORLD frame.

    pts    : a scan (base frame, N x 3), e.g. fresh_scan()
    guess  : world (x, y) of the object guess
    radius : [m] keep points within this horizontal distance of the guess
    Returns an M x 3 array of world points between 3 cm and 45 cm above the floor (the floor and
    anything above the box are dropped)."""
    w = to_world(pts, stretch.pose3d())
    m = (np.hypot(w[:, 0] - guess[0], w[:, 1] - guess[1]) < radius) & (w[:, 2] > 0.03) & (w[:, 2] < 0.45)
    return w[m]


def inspect_candidate(cx, cy):
    """TODO: decide whether the candidate at (cx, cy) is the pick object (OBJECT_SIZE).
    Return (x, y, yaw, points) if it is, else None."""
    assert False, 'TODO (PART 4): inspect_candidate is not implemented yet'


def detect_object():
    """Find the pick object: inspect your object_candidates() one by one with your
    inspect_candidate() (the first 8 if you built the 3-D map, else all of them).

    Returns (x, y, yaw, points) of the first candidate that is the object, or None."""
    cands = object_candidates()
    if vmap.top() is not None:
        cands = cands[:8]                # with heights the box is among the first few
    for cx, cy in cands:
        found = inspect_candidate(cx, cy)
        if found is not None:
            log(f'object found at ({found[0]:.3f}, {found[1]:.3f}), yaw {math.degrees(found[2]):.1f} deg')
            return found
    return None


# =====================================================================================
# PART 5: manipulation - inverse kinematics (base pose + lift + arm) and pick and place
# =====================================================================================
def base_pose_for(target_xy, u, arm_ext):
    """TODO: base pose (x, y, theta) that puts the grasp center on target_xy, with the arm
    pointing along the world unit vector u and extended by arm_ext."""
    assert False, 'TODO (PART 5): base_pose_for is not implemented yet'


def choose_base_pose(target_xy, directions, arm_options=(0.25, 0.30, 0.20, 0.35, 0.15, 0.40, 0.45),
                     check_sweep=True):
    """TODO: best reachable, collision-free base pose to reach target_xy with the arm along one
    of the unit vectors in directions. Return (cost, x, y, theta, arm_ext) or None."""
    assert False, 'TODO (PART 5): choose_base_pose is not implemented yet'


def drive_straight(dist, v=0.05):
    """Drive the base straight along its current heading by `dist` [m] (negative = backward), at most
    `v` [m/s], using the GPS to stop within ~4 mm. For the final few cm when docking."""
    x0, y0, th0 = stretch.pose()
    t0 = robot.getTime()
    while robot.getTime() - t0 < 10:
        x, y, th = stretch.pose()
        err = dist - ((x - x0) * math.cos(th0) + (y - y0) * math.sin(th0))
        if abs(err) < 0.004:
            break
        stretch.drive(float(np.clip(1.5 * err, -v, v)), 2.0 * wrap(th0 - th))
        step()
    stretch.stop()
    wait(0.3)


def dock_at(target_xy, u, arm_ext, measure=False):
    """TODO: drive to the base pose for target_xy and line up precisely.
    Return the arm extension to use, or None if the robot could not get there."""
    assert False, 'TODO (PART 5): dock_at is not implemented yet'


def stow(lift=LIFT_TRAVEL):
    """Make the robot safe to drive: retract the arm, put the lift at `lift` [m] and tuck the
    gripper above the base (WRIST_STOW). Waits until done."""
    move_arm(lift=lift, arm=0.0)
    move_arm(yaw=WRIST_STOW)


def pick(obj):
    """TODO: pick up the object obj = (x, y, yaw, points). Return True if it is held."""
    assert False, 'TODO (PART 5): pick is not implemented yet'


def place():
    """TODO: put the held object on the green pad at GOAL_PAD. Return True when done."""
    assert False, 'TODO (PART 5): place is not implemented yet'


# =====================================================================================
# PROVIDED: map display (400 x 400) and map pictures
# =====================================================================================
_disp = {}


def draw_map():
    """Draw the map on the robot's display (top left of the 3-D view): white = free, gray = unknown,
    black = obstacle (with a 3-D map: orange = low, brown = medium, black = tall), blue = current
    path (viz['path']), green = goals (viz['targets']) and the goal pad, magenta = detected object,
    red = robot. Called automatically about twice a second."""
    D = stretch.display
    if D is None:
        return
    W, H = D.getWidth(), D.getHeight()
    if not _disp:
        s = max(vmap.nx / W, vmap.ny / H)
        _disp.update(s=s, ix=np.clip((np.arange(W) * s).astype(int), 0, vmap.nx - 1),
                     iy=np.clip(((H - 1 - np.arange(H)) * s).astype(int), 0, vmap.ny - 1))
    occ, top = vmap.occupied(), vmap.top()
    img = np.full((vmap.ny, vmap.nx, 3), 110, np.uint8)
    img[vmap.free_count > 0] = (235, 235, 235)
    if top is None:                                      # 2-D map only
        img[occ] = (20, 20, 20)
    else:
        img[occ & (top < 0.4)] = (240, 140, 30)
        img[occ & (top >= 0.4) & (top < 1.0)] = (150, 80, 40)
        img[occ & (top >= 1.0)] = (20, 20, 20)
    out = img[_disp['iy']][:, _disp['ix']].copy()

    def dot(x, y, col, r=2):
        cx, cy = int((x - vmap.x0) / VOX / _disp['s']), int(H - 1 - (y - vmap.y0) / VOX / _disp['s'])
        out[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = col
    for t in viz['targets'] or []:
        dot(*vmap.center(t[2], t[3]), (30, 160, 30), 2)
    for p in (viz['path'] or [])[::2]:
        dot(p[0], p[1], (40, 90, 230), 1)
    if viz['obj'] is not None:
        dot(viz['obj'][0], viz['obj'][1], (230, 0, 200), 4)
    dot(GOAL_PAD[0], GOAL_PAD[1], (0, 170, 60), 4)
    x, y, th = stretch.pose()
    dot(x, y, (230, 30, 30), 4)
    for k in range(1, 6):
        dot(x + 0.06 * k * math.cos(th), y + 0.06 * k * math.sin(th), (230, 30, 30), 1)
    ir = D.imageNew(out.tobytes(), Display.RGB, W, H)
    D.imagePaste(ir, 0, 0, False)
    D.imageDelete(ir)
    D.setColor(0x202020)
    D.drawText(viz['phase'], 4, 4)


def save_maps():
    """Save map.npy (2-D: -1 unknown / 0 free / 1 occupied, rows = y, columns = x) and, if you built
    the 3-D map, map3d.npy (voxel hit counts). Called by main() after the mapping tour."""
    np.save('map.npy', np.where(vmap.occupied(), 1, np.where(vmap.free_count > 0, 0, -1)).astype(np.int8))
    log('map saved: map.npy (-1 unknown / 0 free / 1 occupied)')
    if vmap.hits.any():
        np.save('map3d.npy', vmap.hits)
        log('3-D map saved: map3d.npy (voxel hit counts)')


def save_map_pictures(obj=None):
    """Save map.png (top view, with the detected object `obj` = (x, y, yaw, ...) if given) and,
    with a 3-D map, map3d.png (voxels colored by height) for your report. Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        occ = vmap.occupied()
        img = np.full(occ.shape, 0.6)
        img[vmap.free_count > 0] = 1.0
        img[occ] = 0.0
        ext = [vmap.x0, vmap.x0 + vmap.nx * VOX, vmap.y0, vmap.y0 + vmap.ny * VOX]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray', origin='lower', extent=ext, vmin=0, vmax=1)
        ax.plot(GOAL_PAD[0], GOAL_PAD[1], 's', color='tab:green', ms=9, label='goal pad')
        if obj is not None:
            ax.plot(obj[0], obj[1], 'o', color='magenta', ms=8, label='detected object')
            ax.arrow(obj[0], obj[1], 0.3 * math.cos(obj[2]), 0.3 * math.sin(obj[2]), color='magenta', width=0.02)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Lab 7 map (from the 3-D lidar)')
        ax.legend(loc='lower left')
        fig.savefig('map.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        v = vmap.voxels()
        if len(v) == 0:
            return                                       # no 3-D map (extra credit)
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(projection='3d')
        z = (v[:, 2] + 0.5) * VOX
        ax.scatter(vmap.x0 + (v[:, 1] + 0.5) * VOX, vmap.y0 + (v[:, 0] + 0.5) * VOX, z, c=z, cmap='viridis',
                   s=0.4, marker='s')
        ax.set_box_aspect((vmap.nx, vmap.ny, vmap.nz * 3))
        ax.view_init(elev=55, azim=-60)
        ax.set_title('Lab 7 3-D map (5 cm voxels, colored by height)')
        fig.savefig('map3d.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
    except Exception as ex:      # plotting is optional
        log('map pictures not written:', ex)


# =====================================================================================
# Main
# =====================================================================================
def main():
    """The mission: stow the arm, mapping tour, save + check the map, find the object, pick it
    (up to 3 tries), carry it to the goal stand and place it."""
    stretch.set_gripper(0.0)
    stow()
    for _ in range(10):
        step()
        sense()
    planner.update()
    explore()
    save_maps()
    check_map()
    for attempt in range(3):
        viz['phase'] = 'looking for the object'
        obj = detect_object()
        if obj is None:
            log('object not found in the map')
            return
        save_map_pictures(obj)
        viz.update(obj=obj, phase='object found: going to grasp it')
        if pick(obj):
            break
    else:
        log('FAILED to pick the object')
        return
    viz.update(phase='carrying to the goal stand', obj=None)
    if not place():
        log('FAILED to place the object')
        return
    back_off(0.05)
    viz.update(phase='done', path=None)
    draw_map()
    log(f'mission complete in {robot.getTime() / 60:.1f} min')


main()
stretch.stop()
while robot.step(TIMESTEP) != -1:
    pass
