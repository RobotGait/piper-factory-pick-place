# piper_factory_demo — AgileX PiPER pick & place (MuJoCo)

Feasibility demo — *"Robotics arm for factory"*: an AgileX PiPER 6-DOF arm
picks pouches and places them onto pose-matched target frames, in a physics
simulation (MuJoCo).

## What it does
- Loads the **official AgileX PiPER MJCF** (`piper_ros/src/piper_description/mujoco_model/piper_description.xml`)
- Builds a **digital-twin station**: a pile of **thin rigid pouches** (14.8×14.6 cm, ~2 mm)
  held by a guide fixture, plus a flat placement surface
- **Vision pick:** a **pile camera** colour-segments the top pouch and **back-projects** its pixel
  centroid to a world (x,y) **pick** point (verified **0 mm error** vs. ground truth in sim)
- **Pose-matched place to specified frames:** **plate-sized green frames** are drawn on a flat
  placement surface (an earlier walled tray was dropped — pouches near a slot overhung its wall —
  for a simple flat staging area), each at its own **(x, y, yaw)**. After picking, the arm **lifts the pouch clear and rotates it to
  the frame's angle _while carrying it_** (closed-loop wrist-roll, joint 6, driven per carry segment)
  so it arrives already oriented — no rotate-in-place at the destination. It then descends, **closes
  the loop on position** (a top-down camera re-centres the pouch on the frame centre) and releases —
  each pouch lands matched to its frame in position and orientation (verified **≤10 mm, <4°**).
  A 90°-periodic wrist branch handles joint-6 limits since the pouch is ~square
- Harvests the arm's **near-vertical configurations** and uses position+orientation
  **damped-least-squares IK** so the tool descends **straight down (90°)** onto the pouch
- **Suction** (toggleable weld between link6 and the top pouch) peels **exactly one**
  pouch off the pile and lays it into its target frame; a **cartesian-interpolated carry**
  keeps the pouch tracking the pad. Verified `disturbed: []` — the pile is never knocked
- Renders an mp4 (eye-in-hand wrist-cam inset **top-right**, pile-vision **top-left**,
  place-cam **bottom-left** showing the green frame + placed pouches) and **numerically verifies**
  each peeled pouch matches its specified frame in position *and* orientation (≤10 mm, <4°)

## Why these choices
- **Thin rigid pouch, not flex cloth:** a deformable-cloth pouch was tried first, but a
  suction-held floppy sheet folds/drapes and lands crumpled; a thin rigid slab peels
  singly and packs neatly, which is what a feasibility demo needs to show. (Deformable
  grasping is a great M2+ topic once the pipeline is solid.)
- **Vertical only in a low near-base zone:** PiPER's ±70° wrist limits straight-down reach
  to world z≈0.06–0.10; the cell is laid out there. A tall carton would need a raised mount.
- **Kinematic arm + full-physics pouch:** the stiff, high-reduction PiPER servos are driven
  by joint position (matches the real arm); each pouch is a free rigid body under contact.

## Run
```bash
MJ=/home/robot/encos/do/GaitOne/99_soft/mjlab/.venv/bin/python   # any python with mujoco>=3.11
MUJOCO_GL=egl $MJ mujoco/pick_place_mj.py --pouches 5 --place 3 --fps 30 --out ../../Videos/demo.mp4
```
Output: `[verify] N/N pouches matched their specified frame (pos<30mm, yaw<5deg)` + the video.

## Notes / next steps
- Grasp is a **multi-cup suction pick** — the correct tool for floppy film pouches;
  a parallel jaw cannot grab a flat film from a stack. Confirm EOAT with AgileX support.
- Delivery form: wrap the same motion logic as a **ROS2** node (AgileX ships ROS2 +
  MuJoCo + Gazebo assets). Isaac Sim is an option for a photoreal client render.
- The `piper_ros` clone also has a **ROS2 Humble + MoveIt** config (built) if the SDK
  path favors Gazebo/MoveIt instead of MuJoCo.
