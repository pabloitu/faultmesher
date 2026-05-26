import numpy as np

import faultmesher
from faultmesher.mesh import SurfaceMesh


# gmsh 2D algorithm codes (see Mesh.Algorithm in gmsh docs)
ALGORITHMS = {
    "mesh_adapt": 1,
    "automatic": 2,
    "delaunay": 5,
    "frontal_delaunay": 6,
    "bamg": 7,
}
DEFAULT_ALGORITHM = "delaunay"
INTERSECTIONS = ("weld", "none")
DEFAULT_INTERSECTIONS = "weld"
TRIANGLE_TYPE = 2


def mesh_surface(sys, algorithm=DEFAULT_ALGORITHM, optimize=True,
                 verbose=False, intersections=DEFAULT_INTERSECTIONS,
                 merge_tolerance=None, write_msh=None):
    """Mesh a FaultSystem with gmsh's OCC kernel.

    Each fault is meshed as an independent ruled surface (top edge from
    the trace, bottom edge a rigid down-dip copy). Faults sharing a
    trace endpoint produce near-duplicate vertices along their shared
    edge.

    Parameters
    ----------
    sys : FaultSystem
    algorithm : str or int
        Gmsh 2D meshing algorithm. Accepts a name from
        ``ALGORITHMS`` or any raw gmsh integer code.
    optimize : bool
        Run the Netgen optimizer after generation.
    verbose : bool
        Let gmsh print to stdout.
    intersections : {"weld", "none"}
        "weld" calls merge_meshes after generation to fuse near-duplicate
        vertices within `merge_tolerance` of each other; "none" leaves
        the meshes independent.
    merge_tolerance : float, optional
        Distance threshold for "weld". Defaults to 0.1 * sys.cl.
    write_msh : str, optional
        Path to also save the raw .msh file (for inspection in gmsh GUI).

    Returns
    -------
    SurfaceMesh
        Combined mesh with per-cell tags set to each triangle's fault name.
    """
    import gmsh

    if intersections not in INTERSECTIONS:
        raise ValueError(
            f"intersections must be one of {INTERSECTIONS}, "
            f"got {intersections!r}"
        )
    algo_code = _algorithm_code(algorithm)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.Algorithm", algo_code)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.add("fault_system")

        point_size = {}
        surf_tags_by_name = []
        for fault in sys.faults:
            tags = _add_fault(gmsh, fault, sys.cl, point_size)
            surf_tags_by_name.append((fault.name, tags))

        gmsh.model.occ.synchronize()
        for pt, cl in point_size.items():
            gmsh.model.mesh.setSize([(0, pt)], cl)
        for i, (name, tags) in enumerate(surf_tags_by_name):
            phys = gmsh.model.addPhysicalGroup(2, tags, tag=i + 1)
            gmsh.model.setPhysicalName(2, phys, name)

        gmsh.model.mesh.generate(2)
        if optimize:
            gmsh.model.mesh.optimize("Netgen")
        if write_msh:
            gmsh.write(write_msh)

        surf = _extract(gmsh)
    finally:
        gmsh.finalize()

    if intersections == "weld":
        tol = merge_tolerance if merge_tolerance is not None else 0.1 * sys.cl
        surf.merge_meshes(tolerance=tol)

    faultmesher.log(
        f"mesh_surface: {surf.n_cells} tris, {surf.n_points} pts, "
        f"{len(sys.faults)} fault(s), algorithm={algorithm}, "
        f"intersections={intersections}"
    )
    return surf


def _add_fault(gmsh, fault, cl, point_size):
    """Build one fault as a ruled OCC surface; return its surface tag(s)."""
    occ = gmsh.model.occ
    trace = fault.resampled_trace()
    off = fault.bottom_offset()
    tz, bz = float(fault.top_depth), float(fault.bottom_depth)

    top_pts, bot_pts = [], []
    for x, y in trace:
        tp = occ.addPoint(float(x), float(y), tz)
        bp = occ.addPoint(float(x + off[0]), float(y + off[1]), bz)
        top_pts.append(tp)
        bot_pts.append(bp)
        point_size[tp] = cl
        point_size[bp] = cl

    if fault.curve_type == "spline":
        top_curves = [occ.addSpline(top_pts)]
        bot_curves = [occ.addSpline(bot_pts)]
    else:
        top_curves = [occ.addLine(top_pts[k], top_pts[k + 1])
                      for k in range(len(top_pts) - 1)]
        bot_curves = [occ.addLine(bot_pts[k], bot_pts[k + 1])
                      for k in range(len(bot_pts) - 1)]

    top_wire = occ.addWire(top_curves)
    bot_wire = occ.addWire(bot_curves)
    result = occ.addThruSections([top_wire, bot_wire],
                                 makeSolid=False, makeRuled=True)
    return [tag for _, tag in result]


def _extract(gmsh):
    """Pull the 2D mesh from gmsh, tagged by physical group."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    ien_parts, name_parts = [], []
    for dim, ptag in gmsh.model.getPhysicalGroups(dim=2):
        name = gmsh.model.getPhysicalName(dim, ptag)
        for surf in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
            elem_types, _, node_lists = gmsh.model.mesh.getElements(dim, int(surf))
            for et, ntags in zip(elem_types, node_lists):
                if et != TRIANGLE_TYPE:
                    continue
                tri = np.asarray(ntags, dtype=np.int64).reshape(-1, 3)
                ien_parts.append(np.vectorize(tag_to_idx.__getitem__)(tri))
                name_parts.append(np.full(len(tri), name, dtype=object))

    if not ien_parts:
        raise RuntimeError(
            "gmsh produced no triangles. Check fault geometry and cl."
        )

    return SurfaceMesh(
        xyz=xyz,
        ien=np.vstack(ien_parts),
        tags=np.concatenate(name_parts),
    )


def _algorithm_code(algorithm):
    if isinstance(algorithm, str):
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"unknown algorithm '{algorithm}'; "
                f"choose from {sorted(ALGORITHMS)} or pass an integer code"
            )
        return ALGORITHMS[algorithm]
    if isinstance(algorithm, (int, np.integer)):
        return int(algorithm)
    raise TypeError(
        f"algorithm must be a name or integer, got {type(algorithm).__name__}"
    )