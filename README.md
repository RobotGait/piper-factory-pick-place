# PiPER factory pick & place — vision-guided, pose-matched (MuJoCo)

A physics-simulation demo of an **AgileX PiPER 6-DOF** arm running a full factory
**pick → place** cycle: a camera finds a pouch on a pile, a suction head peels off exactly
one, and the arm lays it onto a target frame **matched in both position and orientation**.

Built in **MuJoCo** (fast, accurate contact physics), packaged as a **ROS2** node.

![demo](docs/demo_pick_place.mp4)

> The demo video is `docs/demo_pick_place.mp4` (top-left = pile camera, bottom-left =
> placement camera, top-right = eye-in-hand wrist camera).

## What it does

- **Vision pick** — a top-down **pile camera** colour-segments the top pouch and
  back-projects its pixel centroid to a world `(x, y)` pick point (**0 mm error** vs.
  ground truth in sim), plus a min-area-rectangle **angle** for the grasp.
- **Single-pouch suction peel** — a **multi-cup vacuum head** (modelled as a toggleable
  weld) comes **straight down (90°)** and peels **exactly one** thin pouch (14.8 × 14.6 cm,
  ~2 mm) off the stack — the pile is never disturbed.
- **Rotate while carrying** — after lifting, the wrist rotates the pouch to the target
  angle **during the transit** (one smooth eased joint-space glide to a pre-solved,
  already-oriented arrival pose — no last-second twist).
- **Pose-matched place** — **plate-sized green target frames** are drawn on a flat surface,
  each at its own `(x, y, yaw)`. A **top-down camera closes the loop on position**, centring
  the pouch on the frame before release. Verified numerically: **3/3 pouches within ≤7 mm
  and <2°** of their specified frame.
- **Smooth motion throughout** — every move is a smoothstep joint-space glide (zero velocity
  at the ends), so chained motions stay jerk-free.

## Key engineering choices

- **Thin rigid pouch, not flex cloth.** A suction-held floppy sheet drapes and lands
  crumpled; a thin rigid slab peels singly and packs neatly — what a feasibility demo needs
  to show. (Deformable grasping is a great follow-up once the pipeline is solid.)
- **Vertical reach zone.** PiPER's ±70° wrist can only point the tool straight down in a
  compact low zone near its base; the cell is laid out there. A tall carton would need the
  arm on a raised mount or short rail.
- **Yaw via joint-6 (wrist roll).** Damped-least-squares IK has little yaw authority at this
  pose, but joint-6 rotates the tool about ~vertical for full yaw control. The pouch is
  ~square, so a 90°-periodic wrist branch handles the joint limits.
- **Kinematic arm + full-physics pouch.** The stiff, high-reduction PiPER servos are driven
  by joint position (matches the real arm); each pouch is a free rigid body under contact.

## Run

```bash
# any Python env with mujoco >= 3.11 and numpy
MUJOCO_GL=egl python piper_factory_demo/mujoco/pick_place_mj.py \
  --pouches 5 --place 3 --fps 30 --out docs/demo_pick_place.mp4
```

Prints `[verify] N/N pouches matched their specified frame (pos<30mm, yaw<5deg)` and writes
the video. It loads the official AgileX PiPER MJCF
([agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros)); point `PIPER_XML`
at your local copy of `piper_description/mujoco_model/piper_description.xml`.

## Layout

```
piper_factory_demo/            ROS2 package
├── mujoco/pick_place_mj.py    the demo (model build, IK, vision, suction, place loop)
├── models/{pouch,box}.sdf     station assets
├── package.xml, setup.py      ROS2 packaging
docs/demo_pick_place.mp4       recorded demo
```

## Roadmap

Robust detection under clutter/lighting · deformable-pouch grasping (the thin 0.17 mm case) ·
separator-board handling · throughput optimization · hardware deployment on the real PiPER.

---

*Sim-only feasibility build. MuJoCo is the physics core; the same motion logic wraps into a
ROS2 node (AgileX ships ROS2 + Gazebo + MuJoCo assets), and a photoreal client render could
target Isaac Sim.*
