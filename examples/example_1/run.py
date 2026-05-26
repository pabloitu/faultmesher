import numpy as np

from faultmesher import Fault, FaultSystem


# trace runs roughly west to east. Walking it in this order, the fault
# dips to the right (south) of the walk direction. Reverse the trace to
# flip the dip side to the north.
trace = np.array([
    [0,    0],
    [3000, 500],
    [6000, 0],
    [9000, 1000],
])

sys_ = FaultSystem(cl=300)
sys_.add(Fault(
    name="fault",
    trace=trace,
    top_depth=0.0,
    bottom_depth=-3000.0,
    dip=70.0,
    curve_type="polyline",
    # spline_resample=200,
))

surf = sys_.build_surface_mesh()
surf.write("fault.vtu")
surf.write("fault.msh")

print(f"{surf.n_cells} triangles, {surf.n_points} nodes")
print(f"  x: {surf.xyz[:, 0].min():.0f} .. {surf.xyz[:, 0].max():.0f}")
print(f"  y: {surf.xyz[:, 1].min():.0f} .. {surf.xyz[:, 1].max():.0f}")
print(f"  z: {surf.xyz[:, 2].min():.0f} .. {surf.xyz[:, 2].max():.0f}")
print(f"  fault names: {surf.fault_names}")