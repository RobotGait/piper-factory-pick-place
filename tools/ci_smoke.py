#!/usr/bin/env python3
"""CI smoke test: run the pick & place demo headless and assert the arm actually succeeded.

The demo prints a line like

    [verify] 3/3 pouches matched their specified frame (pos<30mm, yaw<5deg)

but always exits 0, so a crash-free run tells us nothing about whether the robot
did its job. This wrapper parses that line and fails the build when any pouch
missed its target frame.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEMO = os.path.join(ROOT, "piper_factory_demo", "mujoco", "pick_place_mj.py")

VERIFY = re.compile(r"\[verify\]\s+(\d+)\s*/\s*(\d+)\s+pouches matched")
VISION = re.compile(r"\[vision \d+\].*err\s+(\d+)mm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pouches", type=int, default=3, help="size of the source pile")
    ap.add_argument("--place", type=int, default=1, help="how many to peel & pack")
    ap.add_argument("--vision-tol-mm", type=int, default=5,
                    help="max allowed pile-camera localisation error")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    if not os.path.exists(DEMO):
        print(f"::error::demo script not found: {DEMO}")
        return 1

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")          # CI runners have no GPU
    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])

    cmd = [sys.executable, "-u", DEMO,
           "--pouches", str(args.pouches),
           "--place", str(args.place),
           "--out", os.path.join(ROOT, "ci_run.mp4")]
    print("running:", " ".join(cmd), f"(MUJOCO_GL={env['MUJOCO_GL']})", flush=True)

    try:
        p = subprocess.run(cmd, env=env, timeout=args.timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.TimeoutExpired:
        print(f"::error::demo did not finish within {args.timeout}s")
        return 1

    out = p.stdout or ""
    print(out)

    if p.returncode != 0:
        print(f"::error::demo exited with code {p.returncode}")
        return 1

    m = VERIFY.search(out)
    if not m:
        print("::error::no [verify] line in demo output - cannot confirm the run succeeded")
        return 1

    matched, total = int(m.group(1)), int(m.group(2))
    print(f"placement: {matched}/{total} pouches matched their target frame")

    failed = False
    if total < args.place:
        print(f"::error::expected {args.place} pouch(es) placed, demo reported {total}")
        failed = True
    if matched != total:
        print(f"::error::{total - matched} pouch(es) missed the target frame")
        failed = True

    errs = [int(v) for v in VISION.findall(out)]
    if errs:
        worst = max(errs)
        print(f"vision: worst pile-camera error {worst}mm over {len(errs)} pick(s)")
        if worst > args.vision_tol_mm:
            print(f"::error::pile camera off by {worst}mm (tolerance {args.vision_tol_mm}mm)")
            failed = True
    else:
        print("::warning::no [vision] lines found - vision path may not have run")

    if failed:
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
