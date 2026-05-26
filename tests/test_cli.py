import os

import numpy as np
import pytest

import faultmesher
from faultmesher import FaultSystem, Fault, SurfaceMesh
from faultmesher.cli import main


@pytest.fixture(autouse=True)
def silence():
    prev = faultmesher.verbose
    faultmesher.verbose = False
    yield
    faultmesher.verbose = prev


# from_yaml / to_yaml round-trip

def test_from_yaml_minimal(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "cl: 300\n"
        "faults:\n"
        "  - name: A\n"
        "    trace: [[0, 0], [1000, 0]]\n"
        "    top_depth: 0\n"
        "    bottom_depth: -500\n"
    )
    sys_ = FaultSystem.from_yaml(p)
    assert sys_.cl == 300
    assert len(sys_.faults) == 1
    f = sys_.faults[0]
    assert f.name == "A"
    assert f.dip == 90.0
    assert f.curve_type == "spline"
    np.testing.assert_allclose(f.trace, [[0, 0], [1000, 0]])


def test_from_yaml_full(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "cl: 250\n"
        "faults:\n"
        "  - name: main\n"
        "    trace: [[0, 0], [5000, 0]]\n"
        "    top_depth: 0\n"
        "    bottom_depth: -6000\n"
        "    dip: 80\n"
        "    curve_type: spline\n"
        "    spline_resample: 100\n"
    )
    sys_ = FaultSystem.from_yaml(p)
    f = sys_.faults[0]
    assert f.dip == 80
    assert f.curve_type == "spline"
    assert f.spline_resample == 100


def test_to_yaml_round_trip(tmp_path):
    p = tmp_path / "cfg.yaml"
    a = Fault(name="A", trace=np.array([[0, 0], [1000, 0]]),
              top_depth=0, bottom_depth=-500, dip=90)
    b = Fault(name="B", trace=np.array([[0, 0], [0, 1000]]),
              top_depth=0, bottom_depth=-800, dip=60,
              spline_resample=50)
    sys_ = FaultSystem(cl=200)
    sys_.add(a)
    sys_.add(b)
    sys_.to_yaml(p)

    loaded = FaultSystem.from_yaml(p)
    assert loaded.cl == 200
    assert [f.name for f in loaded.faults] == ["A", "B"]
    assert loaded.faults[1].dip == 60
    assert loaded.faults[1].spline_resample == 50


def test_from_yaml_missing_cl(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("faults: []\n")
    with pytest.raises(ValueError, match="cl"):
        FaultSystem.from_yaml(p)


def test_from_yaml_empty_faults(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("cl: 100\nfaults: []\n")
    with pytest.raises(ValueError, match="non-empty"):
        FaultSystem.from_yaml(p)


def test_from_yaml_missing_required_field(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "cl: 100\n"
        "faults:\n"
        "  - name: A\n"
        "    trace: [[0, 0], [1, 0]]\n"
        "    top_depth: 0\n"
    )
    with pytest.raises(ValueError, match="bottom_depth"):
        FaultSystem.from_yaml(p)


def test_from_yaml_unknown_field(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "cl: 100\n"
        "faults:\n"
        "  - name: A\n"
        "    trace: [[0, 0], [1, 0]]\n"
        "    top_depth: 0\n"
        "    bottom_depth: -10\n"
        "    rake: 90\n"
    )
    with pytest.raises(ValueError, match="unknown"):
        FaultSystem.from_yaml(p)


# CLI

def write_simple_yaml(path):
    path.write_text(
        "cl: 300\n"
        "faults:\n"
        "  - name: A\n"
        "    trace: [[0, 0], [1000, 0]]\n"
        "    top_depth: 0\n"
        "    bottom_depth: -500\n"
    )


def test_cli_run_defaults_next_to_config(tmp_path):
    cfg = tmp_path / "myscene.yaml"
    write_simple_yaml(cfg)
    rc = main(["run", str(cfg), "-q"])
    assert rc == 0
    assert (tmp_path / "myscene.vtu").exists()


def test_cli_run_explicit_dir(tmp_path):
    cfg = tmp_path / "scene.yaml"
    out = tmp_path / "out"
    write_simple_yaml(cfg)
    rc = main(["run", str(cfg), "-o", str(out), "-q"])
    assert rc == 0
    assert (out / "scene.vtu").exists()


def test_cli_run_explicit_filename(tmp_path):
    cfg = tmp_path / "scene.yaml"
    write_simple_yaml(cfg)
    rc = main(["run", str(cfg), "-o", str(tmp_path / "result.msh"), "-q"])
    assert rc == 0
    assert (tmp_path / "result.msh").exists()


def test_cli_run_format_msh(tmp_path):
    cfg = tmp_path / "scene.yaml"
    out = tmp_path / "out"
    write_simple_yaml(cfg)
    rc = main(["run", str(cfg), "-o", str(out), "-f", "msh", "-q"])
    assert rc == 0
    assert (out / "scene.msh").exists()


def test_cli_missing_config_returns_error(tmp_path, capsys):
    rc = main(["run", str(tmp_path / "missing.yaml"), "-q"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cli_invalid_yaml_returns_error(tmp_path, capsys):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("cl: 100\nfaults: []\n")
    rc = main(["run", str(cfg), "-q"])
    assert rc == 1


def test_cli_no_weld_keeps_duplicate_vertices(tmp_path):
    cfg = tmp_path / "scene.yaml"
    out = tmp_path / "out"
    cfg.write_text(
        "cl: 200\n"
        "faults:\n"
        "  - name: A\n"
        "    trace: [[0, 0], [1000, 0]]\n"
        "    top_depth: 0\n"
        "    bottom_depth: -500\n"
        "  - name: B\n"
        "    trace: [[1000, 0], [1000, 1000]]\n"
        "    top_depth: 0\n"
        "    bottom_depth: -500\n"
    )

    main(["run", str(cfg), "-o", str(out), "-q"])
    welded = SurfaceMesh.read(str(out / "scene.vtu"))

    out2 = tmp_path / "out2"
    main(["run", str(cfg), "-o", str(out2), "--no-weld", "-q"])
    unwelded = SurfaceMesh.read(str(out2 / "scene.vtu"))

    assert unwelded.n_points > welded.n_points