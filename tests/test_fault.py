import warnings

import numpy as np
import pytest

from faultmesher import Fault, FaultSystem


# Fault construction and validation

def test_fault_basic():
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-1000)
    assert f.name == "A"
    assert f.curve_type == "spline"
    np.testing.assert_allclose(f.trace[0], [0, 0])
    np.testing.assert_allclose(f.trace[-1], [1000, 0])


def test_fault_rejects_bad_trace():
    with pytest.raises(ValueError, match="shape"):
        Fault(name="A", trace=np.array([0, 0, 1000, 0]))
    with pytest.raises(ValueError, match="at least 2"):
        Fault(name="A", trace=np.array([[0, 0]]))


def test_fault_rejects_bad_curve_type():
    with pytest.raises(ValueError, match="curve_type"):
        Fault(name="A", trace=np.array([[0, 0], [1, 0]]), curve_type="bezier")


def test_fault_rejects_bad_depth_order():
    with pytest.raises(ValueError, match="bottom_depth"):
        Fault(name="A", trace=np.array([[0, 0], [1, 0]]),
              top_depth=-1000, bottom_depth=0)


def test_fault_rejects_bad_dip():
    with pytest.raises(ValueError, match="dip"):
        Fault(name="A", trace=np.array([[0, 0], [1, 0]]),
              top_depth=0, bottom_depth=-1000, dip=120)


# dip direction inference

def test_dip_direction_vertical_is_none():
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-1000, dip=90)
    assert f.dip_direction() is None


def test_dip_direction_east_trace_goes_south():
    # walking +x; right is -y
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-1000, dip=60)
    d = f.dip_direction()
    np.testing.assert_allclose(d, [0, -1], atol=1e-9)


def test_dip_direction_north_trace_goes_east():
    # walking +y; right is +x
    f = Fault(name="A", trace=np.array([[0, 0], [0, 1000]]),
              top_depth=0, bottom_depth=-1000, dip=60)
    d = f.dip_direction()
    np.testing.assert_allclose(d, [1, 0], atol=1e-9)


def test_dip_direction_flips_when_trace_reversed():
    forward = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
                    top_depth=0, bottom_depth=-1000, dip=60)
    reverse = Fault(name="A", trace=np.array([[1000, 0], [0, 0]]),
                    top_depth=0, bottom_depth=-1000, dip=60)
    np.testing.assert_allclose(forward.dip_direction(),
                               -reverse.dip_direction(), atol=1e-9)


def test_dip_direction_is_unit_length():
    f = Fault(name="A", trace=np.array([[0, 0], [1000, 500]]),
              top_depth=0, bottom_depth=-1000, dip=72)
    assert np.linalg.norm(f.dip_direction()) == pytest.approx(1.0)


# bottom offset

def test_bottom_offset_vertical_is_zero():
    f = Fault(name="A", trace=np.array([[0, 0], [1, 0]]),
              top_depth=0, bottom_depth=-1000, dip=90)
    np.testing.assert_allclose(f.bottom_offset(), [0, 0])


def test_bottom_offset_dip_45_east_trace():
    # +x trace, RHR dip is -y, 45 deg, depth 1000 -> bottom is at y=-1000
    f = Fault(name="A", trace=np.array([[0, 0], [1, 0]]),
              top_depth=0, bottom_depth=-1000, dip=45)
    np.testing.assert_allclose(f.bottom_offset(), [0, -1000], atol=1e-10)


def test_bottom_offset_scales_with_depth_extent():
    # vertical extent 5000, dip 30 -> horizontal = 5000 / tan(30)
    f = Fault(name="A", trace=np.array([[0, 0], [0, 1]]),
              top_depth=-1000, bottom_depth=-6000, dip=30)
    expected = 5000.0 / np.tan(np.deg2rad(30))
    # +y trace, RHR dip is +x
    np.testing.assert_allclose(f.bottom_offset(), [expected, 0], atol=1e-9)


# spline resampling

def test_resampled_trace_default_is_unchanged():
    trace = np.array([[0, 0], [1000, 200], [2000, 0]])
    f = Fault(name="A", trace=trace, top_depth=0, bottom_depth=-100)
    np.testing.assert_array_equal(f.resampled_trace(), trace)


def test_resampled_trace_densifies():
    f = Fault(name="A",
              trace=np.array([[0, 0], [10000, 0]]),
              top_depth=0, bottom_depth=-100,
              spline_resample=500)
    rt = f.resampled_trace()
    assert len(rt) > 10
    np.testing.assert_allclose(rt[0], [0, 0])
    np.testing.assert_allclose(rt[-1], [10000, 0])
    gaps = np.linalg.norm(np.diff(rt, axis=0), axis=1)
    assert gaps.max() <= 500 + 1e-9


def test_resampled_trace_follows_polyline_at_bends():
    f = Fault(name="A",
              trace=np.array([[0, 0], [1000, 0], [1000, 1000]]),
              top_depth=0, bottom_depth=-100,
              spline_resample=100)
    rt = f.resampled_trace()
    on_seg = (np.isclose(rt[:, 1], 0) & (rt[:, 0] <= 1000 + 1e-9)) \
           | (np.isclose(rt[:, 0], 1000) & (rt[:, 1] <= 1000 + 1e-9))
    assert on_seg.all()


def test_spline_resample_rejected_for_polyline():
    with pytest.raises(ValueError, match="spline_resample"):
        Fault(name="A", trace=np.array([[0, 0], [1, 0]]),
              curve_type="polyline", spline_resample=10)


def test_spline_resample_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        Fault(name="A", trace=np.array([[0, 0], [1, 0]]),
              spline_resample=0)
    with pytest.raises(ValueError, match="positive"):
        Fault(name="A", trace=np.array([[0, 0], [1, 0]]),
              spline_resample=-5)


# strike deviation warning

def test_warns_on_bent_trace():
    # L-shaped trace, ~90 deg bend
    with pytest.warns(RuntimeWarning, match="bend"):
        Fault(name="A", trace=np.array([[0, 0], [1000, 0], [1000, 1000]]),
              top_depth=0, bottom_depth=-100, dip=60)


def test_no_warning_on_gentle_bend():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Fault(name="A",
              trace=np.array([[0, 0], [1000, 0], [2000, 100]]),
              top_depth=0, bottom_depth=-100, dip=60)


def test_no_warning_for_vertical_even_on_sharp_bend():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Fault(name="A", trace=np.array([[0, 0], [1000, 0], [1000, 1000]]),
              top_depth=0, bottom_depth=-100, dip=90)


# FaultSystem

def make_fault(name, trace, **kw):
    defaults = dict(top_depth=0.0, bottom_depth=-1000.0)
    defaults.update(kw)
    return Fault(name=name, trace=np.asarray(trace, dtype=float), **defaults)


def test_system_add_and_duplicate_name():
    sys_ = FaultSystem()
    sys_.add(make_fault("A", [[0, 0], [1, 0]]))
    with pytest.raises(ValueError, match="duplicate"):
        sys_.add(make_fault("A", [[0, 0], [1, 0]]))


def test_system_cl_must_be_positive():
    with pytest.raises(ValueError, match="cl"):
        FaultSystem(cl=-100)


def test_system_empty_mesh_raises():
    sys_ = FaultSystem()
    with pytest.raises(ValueError, match="no faults"):
        sys_.build_surface_mesh()
