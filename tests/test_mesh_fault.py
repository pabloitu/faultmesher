import numpy as np
import pytest

import faultmesher
from faultmesher import Fault, FaultSystem, SurfaceMesh, mesh_surface


@pytest.fixture(autouse=True)
def silence():
    prev = faultmesher.verbose
    faultmesher.verbose = False
    yield
    faultmesher.verbose = prev


def make_system(faults, cl=500.0):
    sys_ = FaultSystem(cl=cl)
    for f in faults:
        sys_.add(f)
    return sys_


# single-fault basics

def test_vertical_fault_meshes_to_expected_extent():
    f = Fault(name="V", trace=np.array([[0, 0], [2000, 0]]),
              top_depth=0, bottom_depth=-1500, dip=90)
    surf = mesh_surface(make_system([f], cl=300))
    assert surf.n_cells > 0
    assert surf.xyz[:, 0].min() >= -1e-6
    assert surf.xyz[:, 0].max() <= 2000 + 1e-6
    assert surf.xyz[:, 2].min() >= -1500 - 1e-6
    assert surf.xyz[:, 2].max() <= 1e-6
    # vertical fault on y=0 plane
    assert np.allclose(surf.xyz[:, 1], 0, atol=1e-6)


def test_dipping_fault_extrudes_in_dip_direction():
    # trace along +y -> RHR dip direction is +x; at 45 deg, depth 1000,
    # the bottom edge sits at x = 1000
    f = Fault(name="D", trace=np.array([[0, 0], [0, 2000]]),
              top_depth=0, bottom_depth=-1000, dip=45)
    surf = mesh_surface(make_system([f], cl=300))
    z = surf.xyz[:, 2]
    x = surf.xyz[:, 0]
    below = z < -1
    np.testing.assert_allclose(x[below] / -z[below], 1.0, atol=1e-3)


def test_polyline_curve_type_works():
    f = Fault(name="P", trace=np.array([[0, 0], [1000, 200], [2000, 0]]),
              top_depth=0, bottom_depth=-800, dip=90, curve_type="polyline")
    surf = mesh_surface(make_system([f], cl=200))
    assert surf.n_cells > 10


def test_polyline_covers_all_trace_segments():
    # regression: addThruSections returns one face per polyline segment;
    # the mesher must consume them all, not just the first
    trace = np.array([[0, 0], [3000, 500], [6000, 0], [9000, 1000]])
    f = Fault(name="P", trace=trace, top_depth=0, bottom_depth=-1000,
              dip=90, curve_type="polyline")
    surf = mesh_surface(make_system([f], cl=300))
    # mesh should span the full along-trace extent
    assert surf.xyz[:, 0].max() >= 9000 - 1
    # all three segments contribute, so the apex at (3000, 500) and the
    # endpoint at (9000, 1000) must both be present
    has_apex = ((np.abs(surf.xyz[:, 0] - 3000) < 1) &
                (np.abs(surf.xyz[:, 1] - 500) < 1)).any()
    has_end = ((np.abs(surf.xyz[:, 0] - 9000) < 1) &
               (np.abs(surf.xyz[:, 1] - 1000) < 1)).any()
    assert has_apex
    assert has_end


def test_spline_curve_type_works():
    f = Fault(name="S", trace=np.array([[0, 0], [1000, 200], [2000, 0]]),
              top_depth=0, bottom_depth=-800, dip=90, curve_type="spline")
    surf = mesh_surface(make_system([f], cl=200))
    assert surf.n_cells > 10


def test_spline_resample_reduces_overshoot():
    # asymmetric 4-point trace: a default cubic spline interpolates the
    # control points but overshoots between them. Resampling pins it down.
    trace = np.array([[0, 0], [3000, 500], [6000, 0], [9000, 1000]])

    f_loose = Fault(name="A", trace=trace, top_depth=0, bottom_depth=-500,
                    dip=90, curve_type="spline")
    f_tight = Fault(name="A", trace=trace, top_depth=0, bottom_depth=-500,
                    dip=90, curve_type="spline", spline_resample=200)

    surf_loose = mesh_surface(make_system([f_loose], cl=200))
    surf_tight = mesh_surface(make_system([f_tight], cl=200))

    top_loose = surf_loose.xyz[np.isclose(surf_loose.xyz[:, 2], 0), :2]
    top_tight = surf_tight.xyz[np.isclose(surf_tight.xyz[:, 2], 0), :2]

    # the loose spline dips below y=0 (overshoot), the tight one stays
    # within the polyline's y range
    assert top_loose[:, 1].min() < -5
    assert top_tight[:, 1].min() >= -1e-6


# tags

def test_tags_present_and_unique_per_fault():
    a = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    b = Fault(name="B", trace=np.array([[0, 2000], [1000, 2000]]),
              top_depth=0, bottom_depth=-500, dip=90)
    surf = mesh_surface(make_system([a, b], cl=200))
    assert set(surf.fault_names) == {"A", "B"}
    assert surf.tag_mask("A").sum() > 0
    assert surf.tag_mask("B").sum() > 0
    assert surf.tag_mask("A").sum() + surf.tag_mask("B").sum() == surf.n_cells


def test_tag_mask_accepts_fault_instance():
    a = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    surf = mesh_surface(make_system([a], cl=200))
    mask = surf.tag_mask(a)
    assert mask.all()


# intersections / welding

def test_two_faults_sharing_endpoint_get_welded():
    # T-junction: A runs along +x, B branches off the end of A
    a = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    b = Fault(name="B", trace=np.array([[1000, 0], [1000, 1000]]),
              top_depth=0, bottom_depth=-500, dip=90)
    surf = mesh_surface(make_system([a, b], cl=150), intersections="weld")
    # at least one welded vertex pair existed at the shared endpoint;
    # count of unique points after welding should be < sum of independent meshes
    indep_a = mesh_surface(make_system([a], cl=150), intersections="none")
    indep_b = mesh_surface(make_system([b], cl=150), intersections="none")
    assert surf.n_points < indep_a.n_points + indep_b.n_points


def test_intersections_none_keeps_duplicates():
    a = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    b = Fault(name="B", trace=np.array([[1000, 0], [1000, 1000]]),
              top_depth=0, bottom_depth=-500, dip=90)
    surf = mesh_surface(make_system([a, b], cl=150), intersections="none")
    welded = mesh_surface(make_system([a, b], cl=150), intersections="weld")
    assert surf.n_points > welded.n_points


def test_unknown_intersections_mode_rejected():
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    with pytest.raises(ValueError, match="intersections"):
        mesh_surface(make_system([f]), intersections="bogus")


# algorithm

@pytest.mark.parametrize("algo",
                         ["mesh_adapt", "automatic", "delaunay",
                          "frontal_delaunay", "bamg"])
def test_algorithm_names_all_produce_a_mesh(algo):
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    surf = mesh_surface(make_system([f]), algorithm=algo)
    assert surf.n_cells > 0


def test_algorithm_integer_code_accepted():
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    surf = mesh_surface(make_system([f]), algorithm=5)
    assert surf.n_cells > 0


def test_algorithm_unknown_name_rejected():
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    with pytest.raises(ValueError, match="unknown algorithm"):
        mesh_surface(make_system([f]), algorithm="bogus")


# I/O

def test_write_msh_and_vtu(tmp_path):
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=80)
    surf = mesh_surface(make_system([f], cl=200))
    msh = tmp_path / "out.msh"
    vtu = tmp_path / "out.vtu"
    surf.write(str(msh))
    surf.write(str(vtu))
    assert msh.exists() and msh.stat().st_size > 0
    assert vtu.exists() and vtu.stat().st_size > 0


def test_vtu_carries_tags(tmp_path):
    import pyvista as pv
    a = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    b = Fault(name="B", trace=np.array([[0, 2000], [1000, 2000]]),
              top_depth=0, bottom_depth=-500, dip=90)
    surf = mesh_surface(make_system([a, b], cl=200), intersections="none")
    path = tmp_path / "tagged.vtu"
    surf.write(str(path))
    g = pv.read(str(path))
    assert "tag" in g.cell_data


def test_vtu_roundtrip_geometry(tmp_path):
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=70)
    surf = mesh_surface(make_system([f], cl=200))
    path = tmp_path / "rt.vtu"
    surf.write(str(path))
    loaded = SurfaceMesh.read(str(path))
    assert loaded.n_cells == surf.n_cells
    assert loaded.n_points == surf.n_points