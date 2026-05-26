verbose = True


def log(msg):
    if verbose:
        print(msg)


from faultmesher.fault import Fault, FaultSystem
from faultmesher.mesh import SurfaceMesh
from faultmesher.gmsh_backend import mesh_surface

__all__ = ["Fault", "FaultSystem", "SurfaceMesh", "mesh_surface",
           "verbose", "log"]