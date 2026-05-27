# faultmesher

Surface meshing of faults for boundary-element models.

Faults are described by a 2D trace, a depth range, and a dip; faultmesher hands the geometry to gmsh and gives back a triangle mesh you can write to disk or feed into a BEM solver. Slab surfaces from gridded depth rasters (Slab2.0, SURF output, DEMs) are supported through a separate fault type.

## Install

```bash
pip install -e .
```

Python 3.10+. Pulls in numpy, scipy, gmsh, meshio, pyvista, pyyaml, rasterio, pyproj, shapely, and scikit-image.

## A small example

```python
import numpy as np
from faultmesher import Fault, FaultSystem

sys_ = FaultSystem(cl=300)
sys_.add(Fault(
    name="fault",
    trace=np.array([[0, 0], [3000, 500], [6000, 0], [9000, 1000]]),
    top_depth=0,
    bottom_depth=-8000,
    dip=70,
))

mesh = sys_.build_surface_mesh()
mesh.write("fault.vtu")
```

Open `fault.vtu` in ParaView.

## How the trace works

The fault dips to the right of the trace walked in the order given (right-hand rule on strike). Reverse the trace to flip the dip side.

For curved or bent traces, use `curve_type="spline"` (the default) for a smooth surface or `curve_type="polyline"` for a faceted one. Splines through unevenly-spaced control points can overshoot; passing `spline_resample=200` (or similar) densifies the trace first and keeps the result close to the polyline you drew.

## Gridded surfaces

For things like subduction interfaces, give faultmesher a raster:

```python
from faultmesher import GriddedFault, FaultSystem

sys_ = FaultSystem(cl=10000)
sys_.add(GriddedFault(
    name="slab",
    source="depth.tif",
    in_crs="EPSG:4326",
    out_crs="EPSG:32719",    # must be projected, in meters
    max_depth=-70000,
    depth_units="km",
))
mesh = sys_.build_surface_mesh()
```

Any rasterio-supported format works. The raster's valid pixels become a polygon, that polygon is meshed, and each vertex z is sampled from the depth field.

## YAML and CLI

For runs that you want to repeat or share, write the scene as YAML:

```yaml
cl: 300
faults:
  - name: fault
    trace: [[0, 0], [3000, 500], [6000, 0], [9000, 1000]]
    top_depth: 0
    bottom_depth: -8000
    dip: 70
```

then:

```bash
faultmesher run scene.yaml
```

That writes `scene.vtu` next to the YAML. `-o some_dir/` or `-o some_file.msh` overrides the destination. `-f npz` produces a tectosaur-compatible archive.

Gridded faults work the same way:

```yaml
cl: 10000
faults:
  - name: slab
    kind: gridded
    source: depth.tif
    in_crs: EPSG:4326
    out_crs: EPSG:32719
    max_depth: -70000
    depth_units: km
```

## Intersections

When two faults share a trace endpoint, each gets its own vertex there. The default `intersections="weld"` fuses near-duplicate vertices after meshing so the result is watertight at the junction. `intersections="none"` leaves them alone.

Faults that cross *under the surface* without sharing a trace endpoint are not handled yet — coming in a later release.

## Tectosaur output

`mesh.write("scene.npz")` produces a numpy archive that maps directly onto tectosaur's `CombinedMesh`:

```python
import numpy as np
from tectosaur import CombinedMesh

data = np.load("scene.npz", allow_pickle=True)
m = CombinedMesh(list(data["names"]), data["pts"], data["tris"],
                 data["bounds"].tolist())
```

Per-fault names are preserved so you can apply boundary conditions by fault.

## What faultmesher won't do

- Volume meshing. Surfaces only.
- Anything beyond the meshing step itself. No BEM solving, no boundary conditions, no post-processing.
- Reprojection on the trace-based faults. If your traces are in lon/lat, project them yourself before passing them in.

## Repository layout

```
faultmesher/
├── faultmesher/
│   ├── fault.py            Fault, FaultSystem, YAML loader
│   ├── gridded.py          GriddedFault and raster helpers
│   ├── gmsh_backend.py     mesh_surface (calls gmsh OCC)
│   ├── mesh.py             SurfaceMesh container and IO
│   └── cli.py              the `faultmesher` command
├── examples/               runnable scripts and YAMLs
└── tests/                  pytest suite
```
