"""Procedural hand meshes: geometry consistent with the DH model and with the generated URDF."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_robot.model.dh import dh_transform
from hand_robot.tasks.common import Scene
from hand_robot.tasks.task2_single_key import DEFAULT_NOTES
from hand_robot.visualization import hand_mesh as hm


@pytest.fixture(scope="module")
def scene():
    return Scene()


@pytest.fixture(scope="module")
def meshes(scene):
    return hm.build_hand_meshes(scene.robot)


def test_all_meshes_are_closed_outward_solids(meshes):
    assert len(meshes) == 26                                   # base, 20 links, 5 tips
    for parts in meshes.values():
        for p in parts:
            assert np.isfinite(p.V).all() and hm.volume(p.V, p.F) > 0
            assert np.allclose(np.linalg.norm(p.N, axis=1), 1.0, atol=1e-6)


def test_link_shells_span_the_dh_segment(scene, meshes):
    dh = scene.robot.dh
    for i in range(3, 21):
        seg = np.array([dh[i].a, 0.0, dh[i].d])
        L = np.linalg.norm(seg)
        shells = [p for p in meshes[f"link{i}"] if p.name.endswith("_shell")]
        if L < 1e-6:
            assert not shells                                   # zero-length hub: only the hinge cylinder
            continue
        V = shells[0].V @ (seg / L)
        assert abs(V.max() - L) < 0.0106 and abs(V.min()) < 0.0106     # end caps add at most one radius (largest: 10.5 mm, thumb)


def test_tip_meshes_end_exactly_at_the_tcp(scene, meshes):
    for b in scene.robot.branches:
        V = np.vstack([p.V for p in meshes[f"tip_{b}"]])
        assert abs(V[:, 0].max()) < 1e-9                         # tip link frame origin = TCP
        assert V[:, 0].min() > -0.03


def test_knuckles_are_inside_the_palm_in_the_ready_posture(scene, meshes):
    q = scene.ready_posture(DEFAULT_NOTES)
    T = scene.robot.forward_kinematics(q)
    p2 = dh = scene.robot.dh[2]
    L2 = T[2] @ np.linalg.inv(dh_transform(p2.a, p2.alpha, p2.d, 0.0))
    palm = [p for p in meshes["link2"] if p.name == "link2_shell"][0]
    eq = ConvexHull(palm.V).equations
    for b, br in scene.robot.branches.items():
        if b == "finger_5":
            continue
        mcp = np.linalg.inv(L2) @ np.r_[T[br[0]][:3, 3], 1.0]      # MCP = end of the metacarpal link
        assert np.all(eq[:, :3] @ mcp[:3] + eq[:, 3] < 0.012), b    # inside the palm, or at most 12 mm outside its envelope


def test_urdf_references_existing_meshes():
    txt = (ROOT / "gazebo" / "urdf" / "roboliszt.urdf").read_text()
    files = re.findall(r'filename="@MESH_DIR@/([^"]+)"', txt)
    assert len(files) == 48 and len(set(files)) == 48
    for f in files:
        assert (ROOT / "gazebo" / "meshes" / f).is_file()
    assert 'name="tip_contact"' in txt
