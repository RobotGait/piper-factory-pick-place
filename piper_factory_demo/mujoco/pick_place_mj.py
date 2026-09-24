#!/usr/bin/env python3
"""
AgileX PiPER — factory pick & place demo in MuJoCo.

Digital-twin of a pouch-packing station:
  * PiPER 6-DOF arm (official AgileX MJCF) with a suction (vacuum) end effector
  * pouch source: a PILE of THIN rigid pouches (14.8 x 14.6 cm, ~2 mm) held
    aligned by a guide fixture
  * packing tray: a shallow open tray in the arm's top-down dexterous zone

Motion:
  The pad descends VERTICALLY (tool +z straight down, 90 deg to the pouch),
  touches the top pouch, the vacuum turns ON (toggleable weld) so exactly one
  pouch peels off the pile, the arm carries it and lowers it into the tray, then
  the vacuum turns OFF and the pouch settles flat.

Physics-accurate contact grasp: each pouch is a full free rigid body; only the
stiff, high-reduction PiPER joints are driven kinematically (matches the real
position-controlled servo arm).

Usage:
  MUJOCO_GL=egl python pick_place_mj.py --pouches 5 --place 3 --out ../../../Videos/x.mp4
"""
import os, sys, argparse, math
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PIPER_XML = os.path.abspath(os.path.join(
    HERE, "..", "..", "piper_ros", "src", "piper_description",
    "mujoco_model", "piper_description.xml"))

# ---- station geometry (meters), robot base at world origin --------------------
# The arm can only point the tool STRAIGHT DOWN in a low, near-base zone
# (world z ~0.06-0.10). Everything the tool touches vertically lives there.
WORK_Z     = 0.060                     # pick surface / tray floor height
PICK_XY    = (0.28, -0.18)             # pouch pile station (in the vertical zone)
BOX_CTR_XY = (0.20,  0.24)             # packing tray center (in the vertical zone)
# SPECIFIED target frames drawn in the tray, each a DIFFERENT (x, y, yaw): the arm
# must lay each pouch to match its frame's pose. All in the arm's near-vertical low zone.
SLOTS = [
    (0.13, 0.27,  30.0),
    (0.28, 0.11, -25.0),
    (0.13, 0.10,   5.0),
]

POUCH = dict(lx=0.148, ly=0.146, lz=0.002, m=0.015)   # thin RIGID pouch (2 mm)
GAP   = 0.006                                           # gap between stacked pouches
# packing tray footprint ~ the real carton (43x35.5 cm); shallow walls so a
# vertical top-down place clears them
TRAY  = dict(L=0.40, W=0.35, H=0.030, wall=0.008)
GUIDE = dict(clear=0.006, wall=0.006, h=0.045)         # low guide keeps the pile aligned
PAD_LEN = 0.024                                         # suction pad past the TCP


# ---------------------------- model construction -------------------------------
def build_model(n_pouches):
    spec = mujoco.MjSpec.from_file(PIPER_XML)
    spec.option.timestep = 0.002
    spec.option.gravity = [0, 0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    try:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
    except Exception:
        pass

    wb = spec.worldbody
    # moderate, non-specular lighting so coloured pouches don't blow out to white
    # (keeps the vision colour-segmentation reliable)
    wb.add_light(pos=[0.4, 0.0, 1.5], dir=[-0.2, 0.0, -1.0],
                 diffuse=[0.55, 0.55, 0.55], specular=[0.0, 0.0, 0.0])
    wb.add_light(pos=[-0.3, -0.5, 1.5], dir=[0.2, 0.3, -1.0],
                 diffuse=[0.35, 0.35, 0.35], specular=[0.0, 0.0, 0.0])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                rgba=[0.55, 0.57, 0.60, 1], name="floor")

    # work surface — VISUAL ONLY (contype 0) so it never fights the robot base
    tbl = wb.add_body(name="table")
    tg = tbl.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.28, 0.34, WORK_Z/2],
                      pos=[0.24, 0.03, WORK_Z/2], rgba=[0.32, 0.34, 0.38, 1],
                      name="worktable")
    tg.contype = 0
    tg.conaffinity = 0

    # TCP + suction pad + eye-in-hand camera, all on link6
    link6 = spec.body("link6")
    s = link6.add_site()
    s.name = "tcp"
    s.pos = [0.0, 0.0, 0.1358 + PAD_LEN]
    s.size = [0.004, 0.004, 0.004]
    s.rgba = [1, 0, 0, 0.0]
    pad = link6.add_geom()
    pad.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    pad.size = [0.022, PAD_LEN/2, 0.0]
    pad.pos = [0.0, 0.0, 0.1358 + PAD_LEN/2]
    pad.rgba = [0.15, 0.15, 0.18, 1]
    pad.contype = 0                    # pad passes through the pile (vacuum, not push)
    pad.conaffinity = 0
    wcam = link6.add_camera()
    wcam.name = "wrist"
    wcam.pos = [0.0, -0.05, 0.06]
    wcam.quat = [0.0, 1.0, 0.0, 0.0]
    wcam.fovy = 58.0

    # fixed top-down vision camera over the pouch pile (for pouch detection)
    px0, py0 = PICK_XY
    tcam = wb.add_camera()
    tcam.name = "topcam"
    tcam.pos = [px0, py0, 0.55]
    tcam.quat = [1.0, 0.0, 0.0, 0.0]   # identity: camera looks along -world z (straight down)
    tcam.fovy = 42.0

    # fixed top-down vision camera over the packing tray (for placement alignment):
    # sees the pouches already stacked so the next drop can be aligned to them
    bx0, by0 = BOX_CTR_XY
    pcam = wb.add_camera()
    pcam.name = "traycam"
    pcam.pos = [bx0, by0, 0.55]
    pcam.quat = [1.0, 0.0, 0.0, 0.0]
    pcam.fovy = 42.0

    # flat placement surface (no walls): the earlier walled tray was smaller than the
    # spread of the target frames, so pouches near an edge overhung a wall — dropped the
    # tray for a simple flat staging area. Neutral blue-gray (NOT brown) so the orange-
    # pouch colour filter never mistakes the surface itself for a pouch.
    bx, by = BOX_CTR_XY
    pad = wb.add_body(name="place_pad", pos=[bx, by, WORK_Z])
    pad.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.19, 0.19, 0.02],
                 pos=[0, -0.05, 0.004 - 0.02], rgba=[0.34, 0.37, 0.44, 1],
                 name="place_pad_g")   # offset -y so all three slots sit with margin

    # SPECIFIED target frames on the placement surface: one green rectangle outline per slot,
    # each at its own (x, y, yaw). The frame is SIZED TO THE PLATE (outer edge = pouch
    # footprint) so a correctly-placed pouch fills it exactly. Visual only (no collision).
    Lx = POUCH["lx"]/2                 # match the plate exactly
    Ly = POUCH["ly"]/2
    bt, bh, bz = 0.003, 0.006, 0.006
    green = [0.95, 0.85, 0.10, 1]   # yellow target frames
    for si, (sx0, sy0, sdeg) in enumerate(SLOTS):
        sr = math.radians(sdeg)
        frame = wb.add_body(name=f"target_frame{si}", pos=[sx0, sy0, WORK_Z],
                            quat=[math.cos(sr/2), 0, 0, math.sin(sr/2)])
        for nm, fx, fy, hx, hy in [
            (f"f{si}_xp",  Lx, 0,   bt, Ly),
            (f"f{si}_xn", -Lx, 0,   bt, Ly),
            (f"f{si}_yp",  0,  Ly,  Lx, bt),
            (f"f{si}_yn",  0, -Ly,  Lx, bt)]:
            g = frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, bh/2],
                               pos=[fx, fy, bz], rgba=green, name=nm)
            g.contype = 0; g.conaffinity = 0

    # pile guide fixture (keeps the stack aligned)
    px, py = PICK_XY
    hx = POUCH["lx"]/2 + GUIDE["clear"]
    hy = POUCH["ly"]/2 + GUIDE["clear"]
    gw, gh = GUIDE["wall"], GUIDE["h"]
    guide = wb.add_body(name="stack_guide", pos=[px, py, WORK_Z])
    guide.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx+gw, hy+gw, 0.02],
                   pos=[0, 0, -0.02], rgba=[0.30, 0.33, 0.38, 1], name="g_floor")
    for nm, gx, gy, sx, sy in [
        ("g_xp",  hx+gw/2, 0,        gw/2, hy+gw),
        ("g_xn", -(hx+gw/2), 0,      gw/2, hy+gw),
        ("g_yp",  0,        hy+gw/2, hx+gw, gw/2),
        ("g_yn",  0,       -(hy+gw/2), hx+gw, gw/2)]:
        guide.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, gh/2],
                       pos=[gx, gy, gh/2], rgba=[0.40, 0.44, 0.50, 1], name=nm)

    # ---- the pile: thin rigid pouches stacked inside the guide ----
    # saturated orange so the top face stays clearly R>G>B under lighting
    # (pale yellow blew out to near-white and defeated colour segmentation)
    cols = [[0.90, 0.45, 0.10, 1], [0.82, 0.38, 0.08, 1]]
    for i in range(n_pouches):
        z = WORK_Z + POUCH["lz"]/2 + 0.001 + i * (POUCH["lz"] + GAP)
        b = wb.add_body(name=f"pouch{i}", pos=[px, py, z])
        b.add_freejoint()
        g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[POUCH["lx"]/2, POUCH["ly"]/2, POUCH["lz"]/2],
                       rgba=cols[i % 2], mass=POUCH["m"], name=f"pouch{i}_g")
        g.friction = [0.4, 0.005, 0.0001]        # low friction → sheets peel singly
        # toggleable suction weld (vacuum grip) — inactive until the pad engages
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = "link6"
        eq.name2 = f"pouch{i}"
        eq.active = False
        eq.solref = [0.006, 1.0]                  # stiff weld: pouch rigidly tracks the
        eq.solimp = [0.98, 0.995, 0.001, 0.5, 2]  # wrist so commanded yaw is realised

    m = spec.compile()
    m.vis.global_.offwidth = 1280
    m.vis.global_.offheight = 720
    return m


# ------------------------- IK (6D damped least squares) ------------------------
def rot_down(yaw):
    """target link6 orientation: tool +z straight down (world -z), yaw about z."""
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return Rz @ flip


def harvest_vertical_seeds(n_samples=200000, tilt_max=14.0):
    """Sample the arm's joint space for near-straight-down tool configs.
    PiPER's wrist (+-70 deg) can only point the tool down in a small low zone;
    these seeds let the IK converge to a genuinely VERTICAL grasp there."""
    spec = mujoco.MjSpec.from_file(PIPER_XML)
    m = spec.compile()
    d = mujoco.MjData(m)
    arm = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)]
    qadr = [m.jnt_qposadr[j] for j in arm]
    rng = np.array([m.jnt_range[j] for j in arm])
    tcp_off = np.array([0.0, 0.0, 0.1358 + PAD_LEN])
    l6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link6")
    rs = np.random.RandomState(0)
    seeds_q, seeds_p = [], []
    for _ in range(n_samples):
        q = rng[:, 0] + (rng[:, 1] - rng[:, 0]) * rs.rand(6)
        for k, a in enumerate(qadr):
            d.qpos[a] = q[k]
        mujoco.mj_kinematics(m, d)
        R = d.xmat[l6].reshape(3, 3)
        p = d.xpos[l6] + R @ tcp_off
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))
        if tilt < tilt_max and p[2] > 0.03 and 0.12 < math.hypot(p[0], p[1]) < 0.55:
            seeds_q.append(q.copy())
            seeds_p.append(p.copy())
    return np.array(seeds_q), np.array(seeds_p)


def solve_ik(m, d, site_id, tgt_pos, tgt_R, arm_dofs, q_init, iters=400, w_ori=0.5):
    """position+orientation damped least squares; keeps best-by-position solution."""
    q = np.array(q_init, float)
    cols = [m.jnt_dofadr[dof] for dof in arm_dofs]
    dwork = mujoco.MjData(m)
    best_q, best_perr = q.copy(), 1e9
    for _ in range(iters):
        dwork.qpos[:] = d.qpos
        for k, dof in enumerate(arm_dofs):
            dwork.qpos[m.jnt_qposadr[dof]] = q[k]
        mujoco.mj_kinematics(m, dwork)
        mujoco.mj_comPos(m, dwork)
        cur = dwork.site_xpos[site_id].copy()
        curR = dwork.site_xmat[site_id].reshape(3, 3).copy()
        perr = tgt_pos - cur
        pn = np.linalg.norm(perr)
        if pn < best_perr:
            best_perr, best_q = pn, q.copy()
        q_err = np.zeros(4); mujoco.mju_mat2Quat(q_err, (tgt_R @ curR.T).flatten())
        aa = np.zeros(3); mujoco.mju_quat2Vel(aa, q_err, 1.0)
        if pn < 1e-3 and np.linalg.norm(aa) < 2e-2:
            best_q, best_perr = q.copy(), pn
            break
        err = np.hstack([perr, w_ori * aa])
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, dwork, jacp, jacr, site_id)
        J = np.vstack([jacp[:, cols], w_ori * jacr[:, cols]])
        lam = 0.06
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
        q += np.clip(dq, -0.2, 0.2)
        for k, dof in enumerate(arm_dofs):
            lo, hi = m.jnt_range[dof]
            q[k] = min(max(q[k], lo), hi)
    return best_q, best_perr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pouches", type=int, default=5)     # size of the source pile
    ap.add_argument("--place", type=int, default=3)        # how many to peel & pack
    ap.add_argument("--out", default=os.path.join(HERE, "pick_place_demo.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    print("[seeds] harvesting near-vertical arm configurations ...")
    seeds_q, seeds_p = harvest_vertical_seeds()
    print(f"[seeds] {len(seeds_q)} vertical-capable configs found")

    m = build_model(args.pouches)
    d = mujoco.MjData(m)

    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    arm_jnt = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)]
    act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    a_arm = [act[f"joint{i}"] for i in range(1, 7)]

    link6_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link6")
    pouch_bid = {i: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"pouch{i}")
                 for i in range(args.pouches)}
    eq_of = {}
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD:
            for i in range(args.pouches):
                if pouch_bid[i] in (m.eq_obj1id[e], m.eq_obj2id[e]):
                    eq_of[i] = e

    def suction_on(i):
        """activate the weld capturing the live relative pose (vacuum grip)."""
        g, p = link6_bid, pouch_bid[i]
        xg, Rg, qg = d.xpos[g].copy(), d.xmat[g].reshape(3, 3).copy(), d.xquat[g].copy()
        xp, qp = d.xpos[p].copy(), d.xquat[p].copy()
        relpos = Rg.T @ (xp - xg)
        qgi = np.zeros(4); mujoco.mju_negQuat(qgi, qg)
        relq = np.zeros(4); mujoco.mju_mulQuat(relq, qgi, qp)
        data = np.zeros(11); data[3:6] = relpos; data[6:10] = relq; data[10] = 1.0
        m.eq_data[eq_of[i]] = data
        d.eq_active[eq_of[i]] = 1

    def suction_off(i):
        d.eq_active[eq_of[i]] = 0
        va = m.jnt_dofadr[m.body_jntadr[pouch_bid[i]]]
        d.qvel[va:va+6] = 0.0                       # release cleanly, drops straight down

    # home pose
    home = np.array([0.0, 1.2, -1.4, 0.0, 0.7, 0.0])
    d.qpos[:6] = home
    for k, ai in enumerate(a_arm):
        d.ctrl[ai] = home[k]
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, 720, 1280)
    wrist_ren = mujoco.Renderer(m, 260, 340)
    wrist_cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    DETW = 300
    det_ren = mujoco.Renderer(m, DETW, DETW)
    topcam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "topcam")
    traycam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "traycam")

    def back_project(cam_id, u, v, W, H, fovy_deg, z_plane):
        """image pixel (u,v) -> world (x,y) on the horizontal plane z=z_plane."""
        cp = d.cam_xpos[cam_id].copy()
        R = d.cam_xmat[cam_id].reshape(3, 3)          # cols: cam x,y,z axes in world
        xax, yax, zax = R[:, 0], R[:, 1], R[:, 2]
        th = math.tan(math.radians(fovy_deg) / 2.0)
        a = W / H
        xc = (2 * (u + 0.5) / W - 1) * th * a
        yc = (1 - 2 * (v + 0.5) / H) * th
        dirw = xc * xax + yc * yax - zax              # camera looks along -z
        t = (z_plane - cp[2]) / dirw[2]
        w = cp + t * dirw
        return np.array([w[0], w[1]])

    last_det = {"pick": None, "tray": None}   # oriented detections for the vision insets

    def _segment(cam_id):
        """render a top-down camera and return the orange-pouch pixel mask."""
        det_ren.update_scene(d, camera=cam_id)
        img = det_ren.render()
        R, G, B = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
        # orange pouch: strong red, mid green, weak blue (R>G>B by a clear margin)
        return (R > 120) & (R > G + 30) & (G > B + 20) & (B < 150)

    def min_area_rect(xs, ys):
        """rotating-calipers-lite: find the pouch's in-plane angle (the pouch is ~square,
        so orientation is 90-periodic) and the rotated-rectangle corners, in pixels.
        Works in a world-aligned frame (X=+u right, Y=-v up) so the angle IS the world yaw."""
        X = xs.astype(float); Y = -ys.astype(float)
        cx, cy = X.mean(), Y.mean()
        Xc, Yc = X - cx, Y - cy
        if len(Xc) > 3000:                       # subsample for speed
            k = np.linspace(0, len(Xc) - 1, 3000).astype(int); Xc, Yc = Xc[k], Yc[k]
        best = None
        for a in np.linspace(-45.0, 45.0, 91):   # 1-degree search
            r = math.radians(a); c, s = math.cos(r), math.sin(r)
            xr = c * Xc + s * Yc; yr = -s * Xc + c * Yc
            area = (xr.max() - xr.min()) * (yr.max() - yr.min())
            if best is None or area < best[0]:
                best = (area, a, xr.min(), xr.max(), yr.min(), yr.max())
        _, a, xmn, xmx, ymn, ymx = best
        r = math.radians(a); c, s = math.cos(r), math.sin(r)
        corners_uv = []
        for px, py in [(xmn, ymn), (xmx, ymn), (xmx, ymx), (xmn, ymx)]:
            Xw = c * px - s * py + cx; Yw = s * px + c * py + cy   # rotate back
            corners_uv.append((int(round(Xw)), int(round(-Yw))))    # (X,Y)->(u,v)
        return math.radians(a), corners_uv

    def _detect(cam_id, z_plane, key):
        """segment -> centroid (world x,y) + in-plane yaw (world rad). Stores overlay."""
        mask = _segment(cam_id)
        ys, xs = np.where(mask)
        if len(xs) < 50:
            last_det[key] = None
            return None
        u, v = xs.mean(), ys.mean()
        yaw, corners = min_area_rect(xs, ys)
        last_det[key] = {"c": (int(u), int(v)), "corners": corners, "yaw": yaw}
        xy = back_project(cam_id, u, v, DETW, DETW, 42.0, z_plane)
        return xy, yaw

    def detect_pouch(z_plane):
        """PICK vision: top pouch on the pile -> (world xy, world yaw)."""
        return _detect(topcam_id, z_plane, "pick")

    def detect_tray(z_plane):
        """PLACE vision: pouches already stacked in the tray -> (world xy, world yaw).
        The next drop is aligned to this centroid AND rotated to this yaw."""
        return _detect(traycam_id, z_plane, "tray")
    cam = mujoco.MjvCamera()
    cam.lookat = [0.24, 0.03, 0.10]
    cam.distance = 1.15
    cam.azimuth = 132
    cam.elevation = -22
    frames = []
    sim_per_frame = max(1, int((1.0 / args.fps) / m.opt.timestep))

    def draw_line(im, p0, p1, col, th=2):
        h, w, _ = im.shape
        n = int(max(abs(p1[0]-p0[0]), abs(p1[1]-p0[1]))) + 1
        us = np.linspace(p0[0], p1[0], n).round().astype(int)
        vs = np.linspace(p0[1], p1[1], n).round().astype(int)
        for du in range(-(th//2), th//2 + 1):
            for dv in range(-(th//2), th//2 + 1):
                uu = np.clip(us+du, 0, w-1); vv = np.clip(vs+dv, 0, h-1)
                im[vv, uu] = col

    def draw_poly(im, pts, col, th=2):
        for i in range(len(pts)):
            draw_line(im, pts[i], pts[(i+1) % len(pts)], col, th)

    def grab_frame():
        renderer.update_scene(d, camera=cam)
        img = renderer.render().copy()
        # eye-in-hand inset (top-right)
        wrist_ren.update_scene(d, camera=wrist_cam)
        w = wrist_ren.render()
        hh, ww, _ = w.shape
        y0, x0 = 12, img.shape[1] - ww - 12
        img[y0-3:y0+hh+3, x0-3:x0+ww+3] = 40
        img[y0:y0+hh, x0:x0+ww] = w
        def vision_inset(cam_id, det, y_top):
            det_ren.update_scene(d, camera=cam_id)
            dv = det_ren.render().copy()
            if det is not None:
                draw_poly(dv, det["corners"], [0, 255, 60], 2)   # ORIENTED box
                cu, cv = det["c"]
                dv[max(0,cv-6):cv+6, max(0,cu-1):cu+2] = [0, 255, 60]
                dv[max(0,cv-1):cv+2, max(0,cu-6):cu+6] = [0, 255, 60]
            dh, dw2, _ = dv.shape
            img[y_top-3:y_top+dh+3, 12-3:12+dw2+3] = 40
            img[y_top:y_top+dh, 12:12+dw2] = dv
            return dh
        # top-left: PICK vision (pile). bottom-left of it: PLACE vision (tray)
        dh = vision_inset(topcam_id, last_det["pick"], 12)
        vision_inset(traycam_id, last_det["tray"], 12 + dh + 10)
        frames.append(img)

    def hold_arm(q6):
        d.qpos[:6] = q6
        d.qvel[:6] = 0.0
        for k, ai in enumerate(a_arm):
            d.ctrl[ai] = q6[k]

    def step_hold(nsteps, capture=True):
        q6 = d.qpos[:6].copy()
        for i in range(nsteps):
            hold_arm(q6)
            mujoco.mj_step(m, d)
            if capture and (i % sim_per_frame == 0):
                grab_frame()

    def vertical_seed(pos):
        dp = seeds_p - np.array(pos)
        j = int(np.argmin(np.einsum("ij,ij->i", dp, dp)))
        return seeds_q[j].copy()

    def goto(pos, settle=300, w_ori=0.5, seed=None, yaw=0.0, lock_j5=None):
        q_init = d.qpos[:6].copy() if seed == "cur" else vertical_seed(pos)
        q, perr = solve_ik(m, d, site_id, np.array(pos), rot_down(yaw),
                           arm_jnt, q_init, w_ori=w_ori)
        if lock_j5 is not None:      # preserve wrist-roll (pouch orientation) — joint 6
            q[5] = lock_j5           # only affects yaw, not TCP position/tilt
        dw = mujoco.MjData(m); dw.qpos[:6] = q
        mujoco.mj_kinematics(m, dw)
        R = dw.site_xmat[site_id].reshape(3, 3)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))
        if perr > 0.02 or tilt > 12:
            print(f"  [IK] tgt {np.round(pos,3)} residual {perr*1000:.0f}mm tilt {tilt:.0f}deg")
        start = d.qpos[:6].copy()
        for i in range(settle):
            a = (i + 1) / settle
            s = a * a * (3 - 2 * a)          # smoothstep: zero velocity at both ends
            hold_arm((1 - s) * start + s * q)
            mujoco.mj_step(m, d)
            if i % sim_per_frame == 0:
                grab_frame()
        hold_arm(q)
        return q

    def smooth_move(q0, q1, steps):
        """One eased joint-space glide from q0 to q1 (smoothstep — starts and ends at
        zero velocity, so chaining several of these stays jerk-free at the joins)."""
        q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
        for i in range(steps):
            a = (i + 1) / steps
            s = a * a * (3 - 2 * a)
            hold_arm((1 - s) * q0 + s * q1)
            mujoco.mj_step(m, d)
            if i % sim_per_frame == 0:
                grab_frame()
        hold_arm(q1)
        return q1.copy()

    def goto_cart(pos, nseg=12, seg_settle=45, w_ori=0.5, lock_j5=None):
        """move the TCP along a straight CARTESIAN line (re-solving IK each segment)
        so a held pouch tracks the pad smoothly with no wild joint-space swing.
        lock_j5 pins wrist-roll (joint 6) so a pouch already rotated to the target
        orientation keeps it through the carry (joint 6 doesn't affect TCP position)."""
        start_tcp = d.site_xpos[site_id].copy()
        target = np.array(pos, float)
        for s in range(1, nseg + 1):
            wp = start_tcp + (target - start_tcp) * (s / nseg)
            q, _ = solve_ik(m, d, site_id, wp, rot_down(0.0), arm_jnt,
                            d.qpos[:6].copy(), w_ori=w_ori, iters=150)
            if lock_j5 is not None:
                q[5] = lock_j5
            startq = d.qpos[:6].copy()
            for i in range(seg_settle):
                a = (i + 1) / seg_settle
                hold_arm((1 - a) * startq + a * q)
                mujoco.mj_step(m, d)
                if i % sim_per_frame == 0:
                    grab_frame()
            hold_arm(q)
        return d.qpos[:6].copy()

    px, py = PICK_XY
    bx, by = BOX_CTR_XY
    Z_TOUCH  = 0.10                 # vertical approach height just above the pile
    Z_CARRY  = 0.15                 # lift/transit height — clears the guide walls (0.105)
    ROT_Z    = 0.095                # low, near-VERTICAL height where the wrist can spin
                                    # the pouch to any yaw (tool tilt <3deg here)

    def pouch_z(i):
        return float(d.xpos[pouch_bid[i]][2])

    def top_pouch(remaining):
        return max(remaining, key=pouch_z)

    def fold90(a):
        """fold an angle (rad) into [-45,45] deg-equivalent — the pouch is ~square."""
        return (a + math.pi/4) % (math.pi/2) - math.pi/4

    def pouch_yaw(i):
        q = d.xquat[pouch_bid[i]]
        y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
        return fold90(y)

    j5lo, j5hi = m.jnt_range[5][0], m.jnt_range[5][1]

    def _target_j5(q5, err):
        """joint-6 value that yaws the pouch by -err. Prefer the DIRECT value; the pouch
        is square (90-periodic), so only if the direct value is out of the joint limit do
        we fall back to a +/-90deg-equivalent branch (nearest the current wrist angle, so
        corrections stay in the same basin)."""
        direct = q5 - err
        if j5lo <= direct <= j5hi:
            return direct
        best = None
        for k in (-2, -1, 1, 2):
            cand = direct + k * (math.pi / 2)
            if j5lo <= cand <= j5hi and (best is None or abs(cand - q5) < abs(best - q5)):
                best = cand
        return best if best is not None else float(np.clip(direct, j5lo, j5hi))

    def _ramp_j5(q, q5_end, steps):
        """SMOOTHLY rotate wrist-roll from its current value to q5_end (smoothstep)."""
        q5_start = q[5]
        for i in range(steps):
            a = (i + 1) / steps
            s = a * a * (3 - 2 * a)
            q[5] = (1 - s) * q5_start + s * q5_end
            hold_arm(q)
            mujoco.mj_step(m, d)
            if i % sim_per_frame == 0:
                grab_frame()
        return q

    def align_pouch_yaw(idx, target_yaw, q, ramp=300):
        """SLOWLY pre-rotate the HELD pouch to the target orientation with wrist-roll
        (joint 6) — one smooth ramp, not a snap — then a short fine-correction ramp.
        Call over FREE SPACE (nothing under the pouch) so it turns cleanly."""
        q = q.copy()
        err = fold90(target_yaw - pouch_yaw(idx))
        q = _ramp_j5(q, _target_j5(q[5], err), ramp)          # slow main rotation
        for _ in range(4):                                    # smooth fine corrections
            err2 = fold90(target_yaw - pouch_yaw(idx))        # (slope not exactly -1)
            if abs(err2) < math.radians(0.8):
                break
            q = _ramp_j5(q, _target_j5(q[5], err2), 60)
        return q

    def _quat2R(q):
        R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)

    def _site_R_at(q6):
        """FK-only orientation of the tool site at an arbitrary arm config (no physics)."""
        dw = mujoco.MjData(m); dw.qpos[:6] = q6; mujoco.mj_kinematics(m, dw)
        return dw.site_xmat[site_id].reshape(3, 3)

    def _solve_j6_for_yaw(q6, target_yaw, R_rel):
        """Analytically pick wrist-roll (joint 6) so the HELD pouch reaches target_yaw at
        arm config q6 — the pouch is rigidly welded at R_rel to the tool, so its predicted
        world yaw is yaw(site_R(q6) @ R_rel). Newton on joint 6 (d(yaw)/d(j6) ~ -1)."""
        q = np.asarray(q6, float).copy()
        for _ in range(8):
            pR = _site_R_at(q) @ R_rel
            cur = fold90(math.atan2(pR[1, 0], pR[0, 0]))
            err = fold90(target_yaw - cur)
            if abs(err) < math.radians(0.15):
                break
            q[5] = _target_j5(q[5], err)
        return q

    def place_smoothly(idx, drop_x, drop_y, target_yaw, release_z):
        """ONE flowing motion: lift is already done; glide the pouch over to the frame
        while the wrist turns (single eased joint-space move to a PRE-SOLVED oriented
        arrival config — no IK-resolve jitter, no rotate-in-place), descend, nail the
        angle with a light real-feedback touch, close the loop on position, release-ready.
        Returns the locked wrist value."""
        q_lift = d.qpos[:6].copy()
        R_rel = d.site_xmat[site_id].reshape(3, 3).T @ _quat2R(d.xquat[pouch_bid[idx]])

        # arrival over the frame at carry height, already oriented to the frame angle
        q_arr, _ = solve_ik(m, d, site_id, np.array([drop_x, drop_y, Z_CARRY]),
                            rot_down(0.0), arm_jnt, q_lift, w_ori=0.3)
        q_arr = _solve_j6_for_yaw(q_arr, target_yaw, R_rel)
        smooth_move(q_lift, q_arr, 300)                 # carry + rotate, one glide

        # descend to the low near-vertical height, keep the orientation
        q_low, _ = solve_ik(m, d, site_id, np.array([drop_x, drop_y, ROT_Z]),
                            rot_down(0.0), arm_jnt, q_arr, w_ori=0.5)
        q_low = _solve_j6_for_yaw(q_low, target_yaw, R_rel)
        smooth_move(q_arr, q_low, 170)

        # light real-feedback correction (weld compliance vs. prediction), eased
        qc = q_low.copy()
        for _ in range(3):
            err = fold90(target_yaw - pouch_yaw(idx))
            if abs(err) < math.radians(0.6):
                break
            qn = qc.copy(); qn[5] = _target_j5(qc[5], err)
            smooth_move(qc, qn, 70); qc = qn
        lock = float(qc[5])

        # lower straight down, wrist locked
        q_rel, _ = solve_ik(m, d, site_id, np.array([drop_x, drop_y, release_z]),
                            rot_down(0.0), arm_jnt, qc, w_ori=0.5)
        q_rel[5] = lock
        smooth_move(qc, q_rel, 170)

        # close the loop on POSITION: glide the pouch centre onto the frame centre
        for _ in range(4):
            px_i, py_i = d.xpos[pouch_bid[idx]][:2]
            ex, ey = drop_x - float(px_i), drop_y - float(py_i)
            if math.hypot(ex, ey) < 0.003:
                break
            tcp = d.site_xpos[site_id].copy()
            qn, _ = solve_ik(m, d, site_id, np.array([tcp[0] + ex, tcp[1] + ey, release_z]),
                            rot_down(0.0), arm_jnt, d.qpos[:6].copy(), w_ori=0.5)
            qn[5] = lock
            smooth_move(d.qpos[:6].copy(), qn, 90)
        return lock

    step_hold(300)   # let the pile settle in the guide
    remaining = set(range(args.pouches))
    placed = []      # (pouch idx, slot_x, slot_y, slot_deg) for per-frame verification
    for j in range(args.place):
        idx = top_pouch(remaining)
        remaining.discard(idx)
        top_z = pouch_z(idx) + POUCH["lz"]/2
        touch_z = top_z + 0.001
        pre = {i: d.xpos[pouch_bid[i]].copy() for i in range(args.pouches)}

        # --- PICK VISION: detect the top pouch's position AND angle; pick aligned ---
        det = detect_pouch(top_z)
        if det is not None:
            (dxy, pick_yaw) = det
            ppx, ppy = float(dxy[0]), float(dxy[1])
            gt = d.xpos[pouch_bid[idx]][:2]
            print(f"[vision {j}] pouch at ({ppx:.3f},{ppy:.3f}) yaw {math.degrees(pick_yaw):+.1f}deg; "
                  f"truth ({gt[0]:.3f},{gt[1]:.3f}); err {np.linalg.norm(dxy-gt)*1000:.0f}mm")
        else:
            ppx, ppy, pick_yaw = px, py, 0.0
            print(f"[vision {j}] no detection — fall back to nominal pick xy")
        release_z = WORK_Z + 0.004 + POUCH["lz"] + 0.004   # single layer per slot

        # 1) descend VERTICALLY onto the pouch
        goto((ppx, ppy, Z_TOUCH))
        goto((ppx, ppy, touch_z), settle=350)
        # 2) vacuum ON — the top pouch sticks to the pad
        suction_on(idx)
        step_hold(40)
        # 3) PEEL straight up and LIFT HIGH, clear of the guide walls, before moving
        #    sideways (orientation relaxed up here — nothing is being touched)
        goto((ppx, ppy, Z_CARRY), settle=350, w_ori=0.15, seed="cur")
        disturbed = [i for i in range(args.pouches)
                     if i != idx and np.linalg.norm(d.xpos[pouch_bid[i]] - pre[i]) > 0.01]
        print(f"[pick {j}] peeled pouch{idx}; other pouches disturbed: {disturbed}")
        # 4) PLACE to THIS pouch's SPECIFIED green frame — a different (x, y, yaw) each
        #    time. The frame pose is the instruction; the arm must match position + angle.
        slot_x, slot_y, slot_deg = SLOTS[j % len(SLOTS)]
        drop_x, drop_y = slot_x, slot_y
        place_yaw = math.radians(slot_deg)
        print(f"[place {j}] target frame {j} at ({drop_x:.3f},{drop_y:.3f}) "
              f"yaw {slot_deg:+.0f}deg — matching pouch to it")
        # 5) ONE smooth flow: glide to the frame while the wrist turns it (arrives
        #    oriented), descend, finalise the angle, centre it on the frame, ready to drop
        lock = place_smoothly(idx, drop_x, drop_y, place_yaw, release_z)
        print(f"[place {j}] pouch rotated to {math.degrees(pouch_yaw(idx)):+.1f}deg "
              f"(target {slot_deg:+.0f}); wrist locked")
        step_hold(150)
        # 8) vacuum OFF — pouch released, settles flat into the frame
        suction_off(idx)
        step_hold(80)
        goto((drop_x, drop_y, Z_CARRY), settle=280, w_ori=0.15, seed="cur")
        placed.append((idx, slot_x, slot_y, slot_deg))

    goto((0.18, 0.0, Z_CARRY), w_ori=0.15, seed="cur")
    step_hold(200)

    # ---- verify each pouch matched ITS frame (pose = position + orientation) ----
    def yaw_of(i):
        q = d.xquat[pouch_bid[i]]
        y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
        return (math.degrees(y) + 45) % 90 - 45      # square -> fold to [-45,45]
    n_ok = 0
    for (i, sx0, sy0, sdeg) in placed:
        p = d.xpos[pouch_bid[i]]
        perr = math.hypot(p[0]-sx0, p[1]-sy0)
        yerr = abs(((yaw_of(i) - sdeg) + 45) % 90 - 45)
        ok = perr < 0.03 and yerr < 5.0
        if ok:
            n_ok += 1
        print(f"  pouch{i} -> frame({sx0:.2f},{sy0:.2f},{sdeg:+.0f}): "
              f"pos err {perr*1000:.0f}mm, yaw err {yerr:.1f}deg {'OK' if ok else ''}")
    print(f"[verify] {n_ok}/{len(placed)} pouches matched their specified frame "
          f"(pos<30mm, yaw<5deg)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_mp4(frames, args.out, args.fps)
    print(f"[done] {len(frames)} frames -> {args.out}")


def write_mp4(frames, path, fps):
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return
    except Exception as e:
        print("imageio failed:", e)
    import subprocess
    h, w, _ = frames[0].shape
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", path],
        stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()


if __name__ == "__main__":
    main()
