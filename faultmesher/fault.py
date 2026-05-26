from dataclasses import dataclass, field
from typing import Optional, Literal, List
import warnings

import numpy as np


# Warn when the fault's trace bends more than this many degrees away from
# the straight start-to-end direction. Past this point a single rigid dip
# vector is a poor model for the patch.
MAX_TRACE_BEND = 20.0


@dataclass
class Fault:
    """A single fault: along-strike control polyline, depth extent, dip.

    The fault dips to the right of the trace walked in the order given
    (right-hand rule on strike). Reverse the trace to flip the dip side.

    Parameters
    ----------
    name : str
        Identifier. Used as the tag on every triangle generated from
        this fault. Must be unique within a FaultSystem.
    trace : (n, 2) array_like
        Control points in the horizontal plane, ordered along-strike.
    top_depth, bottom_depth : float
        z of the top and bottom edges. bottom_depth must be strictly
        less than top_depth.
    dip : float
        Dip angle in degrees from horizontal. 90 = vertical.
    curve_type : {"spline", "polyline"}
        How to interpret the control points.
    spline_resample : float, optional
        Only with curve_type="spline". Linearly resamples the trace at
        this spacing before feeding it to the spline. Smaller values
        keep the curve closer to the original polyline.
    """

    name: str
    trace: np.ndarray
    top_depth: float = 0.0
    bottom_depth: float = -15000.0
    dip: float = 90.0
    curve_type: Literal["spline", "polyline"] = "spline"
    spline_resample: Optional[float] = None

    def __post_init__(self):
        self.trace = np.ascontiguousarray(self.trace, dtype=float)
        if self.trace.ndim != 2 or self.trace.shape[1] != 2:
            raise ValueError(
                f"Fault {self.name!r}: trace must be shape (n, 2), "
                f"got {self.trace.shape}"
            )
        if len(self.trace) < 2:
            raise ValueError(
                f"Fault {self.name!r}: need at least 2 control points"
            )

        if self.bottom_depth >= self.top_depth:
            raise ValueError(
                f"Fault {self.name!r}: bottom_depth ({self.bottom_depth}) "
                f"must be less than top_depth ({self.top_depth})"
            )
        if not 0 < self.dip <= 90:
            raise ValueError(
                f"Fault {self.name!r}: dip must be in (0, 90], got {self.dip}"
            )

        if self.curve_type not in ("spline", "polyline"):
            raise ValueError(
                f"Fault {self.name!r}: curve_type must be 'spline' or 'polyline'"
            )
        if self.spline_resample is not None:
            if self.curve_type != "spline":
                raise ValueError(
                    f"Fault {self.name!r}: spline_resample only applies "
                    f"when curve_type='spline'"
                )
            if self.spline_resample <= 0:
                raise ValueError(
                    f"Fault {self.name!r}: spline_resample must be positive, "
                    f"got {self.spline_resample}"
                )

        if self.dip != 90.0 and len(self.trace) > 2:
            self._warn_if_too_bent()

    def _warn_if_too_bent(self):
        seg = np.diff(self.trace, axis=0)
        seg_unit = seg / np.linalg.norm(seg, axis=1, keepdims=True)
        walk = self.trace[-1] - self.trace[0]
        walk_unit = walk / np.linalg.norm(walk)
        cos = np.clip(seg_unit @ walk_unit, -1, 1)
        max_bend = np.rad2deg(np.arccos(cos)).max()
        if max_bend > MAX_TRACE_BEND:
            warnings.warn(
                f"Fault {self.name!r}: trace bends up to {max_bend:.1f} deg "
                f"away from the straight start-to-end direction. A single "
                f"rigid dip vector may give a poor fit; consider splitting "
                f"the trace into multiple faults.",
                RuntimeWarning, stacklevel=3,
            )

    def dip_direction(self):
        """Horizontal unit vector in the down-dip direction.

        Perpendicular to the straight start-to-end direction of the trace,
        on its right side (RHR on strike). None for vertical faults.
        """
        if self.dip == 90.0:
            return None
        walk = self.trace[-1] - self.trace[0]
        walk /= np.linalg.norm(walk)
        return np.array([walk[1], -walk[0]])

    def bottom_offset(self):
        """Horizontal (dx, dy) shift from top-edge to bottom-edge."""
        if self.dip == 90.0:
            return np.zeros(2)
        depth = self.top_depth - self.bottom_depth
        return depth / np.tan(np.deg2rad(self.dip)) * self.dip_direction()

    def resampled_trace(self):
        """Trace as the mesher should consume it.

        When spline_resample is set, the polyline is linearly interpolated
        so consecutive points sit no further apart than spline_resample.
        Endpoints are always preserved. Otherwise the original trace is
        returned unchanged.
        """
        if self.spline_resample is None:
            return self.trace
        seg_len = np.linalg.norm(np.diff(self.trace, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        n_new = int(np.ceil(cum[-1] / self.spline_resample)) + 1
        s = np.linspace(0.0, cum[-1], n_new)
        return np.column_stack([
            np.interp(s, cum, self.trace[:, 0]),
            np.interp(s, cum, self.trace[:, 1]),
        ])


@dataclass
class FaultSystem:
    """A collection of faults sharing a target edge length.

    Faults are meshed independently. If two faults share a trace endpoint,
    the resulting meshes will have near-duplicate vertices along the shared
    edge; call SurfaceMesh.merge_meshes (or pass intersections="weld" to
    mesh_surface) to weld them.

    Parameters
    ----------
    cl : float
        System-wide target edge length.
    """

    cl: float = 2000.0
    faults: List[Fault] = field(default_factory=list)

    def __post_init__(self):
        if self.cl <= 0:
            raise ValueError(f"cl must be positive, got {self.cl}")

    def add(self, fault):
        if any(f.name == fault.name for f in self.faults):
            raise ValueError(f"duplicate fault name: {fault.name!r}")
        self.faults.append(fault)
        return fault

    def build_surface_mesh(self, **kwargs):
        if not self.faults:
            raise ValueError("FaultSystem has no faults to mesh")
        from faultmesher.gmsh_backend import mesh_surface
        return mesh_surface(self, **kwargs)

    @classmethod
    def from_yaml(cls, path):
        """Load a FaultSystem from a YAML file.

        Expected layout::

            cl: 300
            faults:
              - name: main
                trace: [[0, 0], [1000, 500], [2000, 0]]
                top_depth: 0
                bottom_depth: -8000
                dip: 70
                curve_type: spline       # optional
                spline_resample: 200     # optional

        ``cl`` and ``faults`` are required; each fault must give at least
        ``name``, ``trace``, ``top_depth`` and ``bottom_depth``. Other
        fields fall back to their Fault dataclass defaults.
        """
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data, source=str(path))

    def to_yaml(self, path):
        """Write the system to a YAML file readable by from_yaml."""
        import yaml
        data = {
            "cl": float(self.cl),
            "faults": [{
                "name": f.name,
                "trace": f.trace.tolist(),
                "top_depth": float(f.top_depth),
                "bottom_depth": float(f.bottom_depth),
                "dip": float(f.dip),
                "curve_type": f.curve_type,
                **({"spline_resample": float(f.spline_resample)}
                   if f.spline_resample is not None else {}),
            } for f in self.faults],
        }
        with open(path, "w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)

    @classmethod
    def _from_dict(cls, data, source="<dict>"):
        if not isinstance(data, dict):
            raise ValueError(f"{source}: top level must be a mapping")
        if "cl" not in data:
            raise ValueError(f"{source}: missing required key 'cl'")
        if "faults" not in data:
            raise ValueError(f"{source}: missing required key 'faults'")
        if not isinstance(data["faults"], list) or not data["faults"]:
            raise ValueError(f"{source}: 'faults' must be a non-empty list")

        sys_ = cls(cl=float(data["cl"]))
        required = ("name", "trace", "top_depth", "bottom_depth")
        allowed = required + ("dip", "curve_type", "spline_resample")
        for i, entry in enumerate(data["faults"]):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{source}: faults[{i}] must be a mapping"
                )
            missing = [k for k in required if k not in entry]
            if missing:
                raise ValueError(
                    f"{source}: faults[{i}] missing key(s): {missing}"
                )
            unknown = [k for k in entry if k not in allowed]
            if unknown:
                raise ValueError(
                    f"{source}: faults[{i}] has unknown key(s): {unknown}"
                )
            sys_.add(Fault(
                name=entry["name"],
                trace=np.asarray(entry["trace"], dtype=float),
                top_depth=float(entry["top_depth"]),
                bottom_depth=float(entry["bottom_depth"]),
                dip=float(entry.get("dip", 90.0)),
                curve_type=entry.get("curve_type", "spline"),
                spline_resample=(float(entry["spline_resample"])
                                 if "spline_resample" in entry else None),
            ))
        return sys_