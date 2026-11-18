"""Procedural visual meshes of the 20-DoF hand (no CAD data available: shapes are designed, dimensions come from the DH model).

Every mesh is expressed in the URDF link frame L_i (origin = joint i axis, z = joint axis; the DH segment of the link is
seg = [a, 0, d] in L_i, from the joint i axis to the joint i+1 axis). Output: ``build_hand_meshes(robot)`` ->
{link_name: [Part(name, material, V, F, N)]}; ``write_obj`` writes Wavefront OBJ with smooth vertex normals.

Look: robot-style hand (Shadow-Hand-like): off-white shells, graphite hinge cylinders on every joint axis, dark silicone
pads on the palmar face of the fingertips, human-like palm (narrow at the wrist, widening towards the knuckle arch, thenar and
hypothenar swellings), wrist cuff and a short forearm on the carriage.
These meshes are VISUAL ONLY. Collisions stay on the fingertips (sphere), inertias come from the rigid-body model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull

MATERIALS = {                      # rgba
    "shell": (0.88, 0.89, 0.91, 1.0),
    "graphite": (0.16, 0.17, 0.19, 1.0),
    "pad": (0.10, 0.11, 0.13, 1.0),
}

TIP_RADIUS = 0.0055                # = piano.tip_collision_radius (the collision sphere has the same size)
FINGER_SCALE = {"finger_1": 0.90, "finger_2": 1.00, "finger_3": 1.05, "finger_4": 1.00, "finger_5": 1.18}
FINGER_THICKNESS = 0.82            # visual slimming of the four fingers (the key pitch is 23.5 mm: thick fingers look glued together)
HINGE_LEN = 1.9                    # hinge cylinder length / finger radius (< 2: the hinge never sticks out sideways)

# Palm frame (link2): x along the hand, y = palm normal (palmar = -y), z lateral. From the DH/URDF convention of
# gazebo/generate.py the thumb root is at z = -44 mm and the little-finger knuckle at z = +37 mm (flip the signs if mirrored).
PALM_RADIAL_SIGN = -1.0
PALM_PALMAR_SIGN = -1.0


@dataclass
class Part:
    name: str
    material: str
    V: np.ndarray
    F: np.ndarray
    N: np.ndarray


# --------------------------------------------------------------------------------------------- primitives
def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])          # area-weighted
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.where(n > 1e-15, n, 1.0)


def _frame(axis: np.ndarray):
    axis = axis / np.linalg.norm(axis)
    h = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, h)
    e1 /= np.linalg.norm(e1)
    return axis, e1, np.cross(axis, e1)


def revolve(p0, axis, profile, n_around=28) -> tuple[np.ndarray, np.ndarray]:
    """Surface of revolution about the axis through p0; profile = [(s, radius)] ordered along the axis (first/last radius 0).
    Triangles are wound outward."""
    axis, e1, e2 = _frame(np.asarray(axis, float))
    th = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    ring = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2               # (n, 3)
    V = np.array([np.asarray(p0) + s * axis + r * ring[j] for s, r in profile for j in range(n_around)])
    nr = len(profile)
    F = []
    for i in range(nr - 1):
        for j in range(n_around):
            a, b = i * n_around + j, i * n_around + (j + 1) % n_around
            c, d = (i + 1) * n_around + j, (i + 1) * n_around + (j + 1) % n_around
            F += [[a, b, c], [b, d, c]]
    F = np.array(F)
    # orient outward: majority of faces must point away from the axis
    ctr = V.mean(axis=0)
    fc = V[F].mean(axis=1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if np.sum(np.einsum("ij,ij->i", fn, fc - ctr)) < 0:
        F = F[:, ::-1]
    return V, F


def tapered_capsule(p0, p1, r0, r1, n_around=28, n_cap=7):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    prof = [(-r0 * np.cos(a), r0 * np.sin(a)) for a in np.linspace(0, np.pi / 2, n_cap)]
    prof += [(L + r1 * np.cos(a), r1 * np.sin(a)) for a in np.linspace(np.pi / 2, 0, n_cap)]
    return revolve(p0, p1 - p0, prof, n_around)


def sphere(center, r, n_around=24, n_lat=12):
    prof = [(-r * np.cos(a), r * np.sin(a)) for a in np.linspace(0, np.pi, n_lat + 1)]
    prof = [(s + r, rad) for s, rad in prof]
    return revolve(np.asarray(center, float) - np.array([0, 0, r]), [0, 0, 1], prof, n_around)


def cylinder(center, axis, r, length, n_around=28, bevel=0.0):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    p0 = np.asarray(center, float) - axis * length / 2
    b = min(bevel, r * 0.5, length * 0.4)
    prof = [(0.0, 0.0), (0.0, r - b), (b, r), (length - b, r), (length, r - b), (length, 0.0)]
    return revolve(p0, axis, prof, n_around)


def ellipsoid(center, radii, n_around=24, n_lat=12):
    V, F = sphere(np.zeros(3), 1.0, n_around, n_lat)
    return V * np.asarray(radii) + np.asarray(center), F


def hull_of_spheres(centers, radii, n_pts=260, seed=0):
    """Smooth rounded convex solid: convex hull of spheres (Fibonacci samples), normals taken from the sphere directions."""
    centers = np.asarray(centers, float)
    k = np.arange(n_pts) + 0.5
    phi, th = np.arccos(1 - 2 * k / n_pts), np.pi * (1 + 5 ** 0.5) * k
    dirs = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    pts = np.vstack([c + r * dirs for c, r in zip(centers, np.broadcast_to(radii, len(centers)))])
    hull = ConvexHull(pts)
    F = hull.simplices.copy()
    fn = np.cross(pts[F[:, 1]] - pts[F[:, 0]], pts[F[:, 2]] - pts[F[:, 0]])
    flip = np.einsum("ij,ij->i", fn, pts[F].mean(axis=1) - pts.mean(axis=0)) < 0
    F[flip] = F[flip][:, ::-1]
    used = np.unique(F)
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used))
    return pts[used], remap[F]


def _part(name, material, VF) -> Part:
    V, F = VF
    return Part(name, material, np.asarray(V, float), np.asarray(F, int), vertex_normals(V, F))


def merge(*VFs):
    V, F, off = [], [], 0
    for v, f in VFs:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F)


def write_obj(path, part: Part):
    with open(path, "w") as fh:
        fh.write(f"# RoboLiszt procedural mesh: {part.name}\no {part.name}\n")
        for v in part.V:
            fh.write("v %.6f %.6f %.6f\n" % tuple(v))
        for n in part.N:
            fh.write("vn %.5f %.5f %.5f\n" % tuple(n))
        for f in part.F + 1:
            fh.write("f %d//%d %d//%d %d//%d\n" % (f[0], f[0], f[1], f[1], f[2], f[2]))


def volume(V, F) -> float:
    """Signed volume (positive for outward-wound closed meshes)."""
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


# --------------------------------------------------------------------------------------------- the palm
def _gauss(x, z, x0, z0, sx, sz):
    return np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((z - z0) / sz) ** 2))


# closed outline (x, z) in "design" coordinates (radial = +z): narrow wrist -> widening ulnar edge (hypothenar) -> arch through the
# knuckle row (little, ring, middle, index; 4-5 mm inside the MCP joints) -> web -> thenar -> wrist
PALM_OUTLINE = np.array([
    [-0.024, -0.024], [-0.007, -0.030], [0.012, -0.040], [0.034, -0.046], [0.052, -0.047], [0.061, -0.042],
    [0.0635, -0.0345], [0.077, -0.0206], [0.090, 0.000], [0.083, 0.0202], [0.072, 0.031], [0.058, 0.038],
    [0.040, 0.044], [0.022, 0.048], [0.003, 0.043], [-0.014, 0.034], [-0.025, 0.024]])


def palm_mesh(n_phi=72, n_t=36, q=3.2):
    """Human-like palm: closed outline inflated into a rounded solid of variable thickness.
    Frame of link2: x along the hand, y thickness (palmar = -y), z lateral (radial = PALM_RADIAL_SIGN * z)."""
    out = PALM_OUTLINE
    tck, _ = splprep([np.r_[out[:, 0], out[0, 0]], np.r_[out[:, 1], out[0, 1]]], s=0, per=True)
    px, pz = splev(np.linspace(0, 1, n_phi, endpoint=False), tck)
    c0 = np.array([0.040, 0.000])
    yc = -0.0055

    def hd(X, Z):                                   # dorsal half-thickness: gently convex back of the hand
        return (0.0092 + 0.0022 * np.clip(1 - ((Z - 0.004) / 0.050) ** 2, 0, 1)
                + 0.0035 * _gauss(X, Z, 0.012, 0.044, 0.018, 0.012))        # swelling over the thumb root

    def hp(X, Z):                                   # palmar half-thickness: swellings and central hollow
        return (0.0100 + 0.0075 * _gauss(X, Z, 0.018, 0.034, 0.018, 0.015)      # thenar eminence
                       + 0.0042 * _gauss(X, Z, 0.032, -0.036, 0.022, 0.011)     # hypothenar eminence
                       - 0.0030 * _gauss(X, Z, 0.052, 0.000, 0.018, 0.016))     # palm hollow

    rings = []
    for t in np.linspace(0, np.pi, n_t)[1:-1]:
        u = np.cos(t)
        r = (1 - abs(u) ** q) ** (1 / q)
        X = c0[0] + r * (px - c0[0])
        Z = c0[1] + r * (pz - c0[1])
        h = hd(X, Z) if u >= 0 else hp(X, Z)
        rings.append(np.c_[X, yc + u * h, Z])
    top = np.array([c0[0], yc + hd(*c0), c0[1]])
    bot = np.array([c0[0], yc - hp(*c0), c0[1]])
    V = np.vstack([top[None]] + rings + [bot[None]])
    V[:, 2] *= PALM_RADIAL_SIGN
    if PALM_PALMAR_SIGN > 0:
        V[:, 1] *= -1
    nr, F = len(rings), []
    ring = lambda i, j: 1 + i * n_phi + (j % n_phi)
    for j in range(n_phi):
        F.append([0, ring(0, j), ring(0, j + 1)])
        F.append([len(V) - 1, ring(nr - 1, j + 1), ring(nr - 1, j)])
        for i in range(nr - 1):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1)
            F += [[a, c, b], [b, c, d]]
    F = np.array(F)
    if volume(V, F) < 0:                            # outward winding
        F = F[:, ::-1]
    return V, F


# --------------------------------------------------------------------------------------------- the hand
def _finger_link_parts(i, seg, rs, re, scale):
    """Shell capsule along seg plus graphite hinge cylinder on the joint axis (z) at the link origin."""
    parts = []
    L = np.linalg.norm(seg)
    if L > 1e-6:
        parts.append(_part(f"link{i}_shell", "shell", tapered_capsule([0, 0, 0], seg, rs, re)))
    parts.append(_part(f"link{i}_joint", "graphite",
                       cylinder([0, 0, 0], [0, 0, 1], rs * 1.08, HINGE_LEN * rs, bevel=0.0012)))
    return parts


def build_hand_meshes(robot) -> dict[str, list[Part]]:
    out: dict[str, list[Part]] = {}
    dh = robot.dh

    def seg_of(i):
        return np.array([dh[i].a, 0.0, dh[i].d])

    # --- carriage / forearm (static, on base_link) and wrist cuff (link 1)
    fore = _part("base_forearm", "graphite", tapered_capsule([-0.22, 0, 0], [-0.015, 0, 0], 0.029, 0.031, n_around=36, n_cap=8))
    ring = _part("base_ring", "shell", cylinder([-0.018, 0, 0], [1, 0, 0], 0.0335, 0.010, n_around=36, bevel=0.0015))
    out["base_link"] = [fore, ring]
    out["link1"] = [_part("link1_shell", "shell", tapered_capsule([0, 0, 0], seg_of(1), 0.0285, 0.0285, n_around=36, n_cap=6))]

    # --- palm (link 2) and wrist hinge
    palm = _part("link2_shell", "shell", palm_mesh())
    wrist = _part("link2_joint", "graphite", cylinder([0, 0, 0], [0, 0, 1], 0.0125, 0.075, bevel=0.002))
    out["link2"] = [palm, wrist]

    # --- fingers and thumb
    th = FINGER_THICKNESS
    for b, br in robot.branches.items():
        s = FINGER_SCALE[b]
        if b != "finger_5":
            bar, rest = br[0], br[1:]
            out[f"link{bar}"] = [_part(f"link{bar}_shell", "shell", tapered_capsule([0, 0, 0], seg_of(bar), 0.0060 * s * th, 0.0078 * s * th)),
                                 _part(f"link{bar}_joint", "graphite", cylinder([0, 0, 0], [0, 0, 1], 0.0075 * s * th, 0.0090 * s * th, bevel=0.001))]
            ph = [j for j in rest if np.linalg.norm(seg_of(j)) > 1e-6]
            r_top, r_end = 0.0072 * s * th, max(0.0062 * th, TIP_RADIUS * 1.05)
            radii = np.linspace(r_top, r_end, len(ph) + 1)
            k = 0
            for j in rest:
                if np.linalg.norm(seg_of(j)) > 1e-6:
                    out[f"link{j}"] = _finger_link_parts(j, seg_of(j), radii[k], radii[k + 1], s)
                    k += 1
                else:                                                  # zero-length hub at the MCP
                    out[f"link{j}"] = _finger_link_parts(j, seg_of(j), r_top, r_top, s)
            tip_r0 = radii[-1]
        else:                                                          # thumb: 16 thenar bar, 17/18 hubs, 19, 20 phalanges
            out["link16"] = [_part("link16_shell", "shell", tapered_capsule([0, 0, 0], seg_of(16), 0.0105, 0.0098)),
                             _part("link16_joint", "graphite", cylinder([0, 0, 0], [0, 0, 1], 0.0105, 0.020, bevel=0.0012))]
            out["link17"] = _finger_link_parts(17, seg_of(17), 0.0098, 0.0098, s)
            out["link18"] = _finger_link_parts(18, seg_of(18), 0.0098, 0.0098, s)
            out["link19"] = _finger_link_parts(19, seg_of(19), 0.0090, 0.0078, s)
            out["link20"] = _finger_link_parts(20, seg_of(20), 0.0078, 0.0066, s)
            tip_r0 = 0.0066
        # distal phalanx (built from the last joint, TCP at +L_TIP along x; shifted below into the tip link frame)
        L = robot.L_TIP
        shell = tapered_capsule([0, 0, 0], [L - TIP_RADIUS, 0, 0], tip_r0, TIP_RADIUS)
        # palmar pad: flexion with q<0 curls +x toward -y, so the contact face is -y
        pad = ellipsoid([L - TIP_RADIUS - 0.0020, -0.0031, 0.0], [0.0072, 0.0030, TIP_RADIUS * 0.80])
        # the tip LINK frame has its origin at the TCP (fixed-joint origin = F_last . Tx(L_TIP)): shift by -L_TIP along x
        shift = np.array([-L, 0.0, 0.0])
        shell, pad = (shell[0] + shift, shell[1]), (pad[0] + shift, pad[1])
        out[f"tip_{b}"] = [_part(f"tip_{b}_shell", "shell", shell), _part(f"tip_{b}_pad", "pad", pad)]
    return out