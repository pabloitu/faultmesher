import numpy as np

from faultmesher import Fault, FaultSystem


sys_ = FaultSystem(cl=250)

sys_.add(Fault(
    name="main",
    trace=np.array([[0, 0], [5000, 0], [10000, 800]]),
    top_depth=0.0,
    bottom_depth=-3000.0,
    dip=80.0,
    curve_type="spline",
))

sys_.add(Fault(
    name="splay",
    trace=np.array([[11000, 800], [13000, 3000]]),
    top_depth=0.0,
    bottom_depth=-2500.0,
    dip=55.0,
    curve_type="polyline",
))

# weld vertices at the shared trace endpoint
surf = sys_.build_surface_mesh(intersections="weld")
surf.write("two_faults.vtu")

print(f"{surf.n_cells} triangles, {surf.n_points} nodes")
for name in surf.fault_names:
    print(f"  {name}: {int(surf.tag_mask(name).sum())} triangles")