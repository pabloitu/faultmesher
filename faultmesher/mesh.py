import os
import warnings
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import faultmesher


class SurfaceMesh:
    """Triangle surface mesh with optional per-cell fault tags."""

    def __init__(self, xyz=None, ien=None, tags=None):
        self.xyz = xyz
        self.ien = ien
        self.tags = tags

    @property
    def n_cells(self):
        return self.ien.shape[0]

    @property
    def n_points(self):
        return self.xyz.shape[0]

    def centroids(self):
        return self.xyz[self.ien].mean(axis=1)

    def areas(self):
        p0 = self.xyz[self.ien[:, 0]]
        p1 = self.xyz[self.ien[:, 1]]
        p2 = self.xyz[self.ien[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    def normals(self):
        """Unit normal of each triangle by right-hand rule on ien."""
        p0 = self.xyz[self.ien[:, 0]]
        p1 = self.xyz[self.ien[:, 1]]
        p2 = self.xyz[self.ien[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return n

    def edge_lengths(self):
        """Edge lengths of every triangle, shape (n_cells, 3)."""
        a = self.xyz[self.ien[:, 0]]
        b = self.xyz[self.ien[:, 1]]
        c = self.xyz[self.ien[:, 2]]
        return np.column_stack([
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ])

    # tag selection

    def tag_mask(self, fault):
        """Boolean mask of triangles belonging to a fault.

        Accepts a Fault instance (uses its name) or a string tag.
        """
        if self.tags is None:
            raise ValueError("this SurfaceMesh has no tags")
        name = fault.name if hasattr(fault, "name") else str(fault)
        mask = self.tags == name
        if not mask.any():
            known = np.unique(self.tags).tolist()
            raise KeyError(f"no triangles tagged {name!r}; known tags: {known}")
        return mask

    @property
    def fault_names(self):
        if self.tags is None:
            return []
        return np.unique(self.tags).tolist()

    # winding

    def fix_winding(self, reference=(1, 1, 1), dry_run=False):
        """Make winding consistent and orient each component outward.

        Triangles are first re-stitched so that each connected component
        has internally consistent winding. Then each component's average
        normal is checked against `reference`; components whose normals
        point into the opposite half-space are flipped wholesale.

        With dry_run=True, the mesh is not modified and the method
        returns the same report as a real call would have produced.

        Returns
        -------
        dict
            n_components, n_inconsistent, n_flipped, ambiguous, components.
        """
        comp, needs_flip_stitch = self._winding_components()
        n_components = comp.max() + 1 if len(comp) else 0

        ien = self.ien if not dry_run else self.ien.copy()
        if needs_flip_stitch.any():
            ien[needs_flip_stitch] = ien[needs_flip_stitch][:, [0, 2, 1]]

        # per-component average normal score against reference
        ref = np.asarray(reference, dtype=float)
        ref = ref / np.linalg.norm(ref)
        # recompute normals from the (possibly stitched) ien
        p0 = self.xyz[ien[:, 0]]
        p1 = self.xyz[ien[:, 1]]
        p2 = self.xyz[ien[:, 2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

        components = []
        n_flipped = 0
        ambiguous = []
        for cid in range(n_components):
            mask = comp == cid
            avg = nrm[mask].mean(axis=0)
            avg /= max(np.linalg.norm(avg), 1e-12)
            score = float(avg @ ref)
            if abs(score) < 0.1:
                ambiguous.append(cid)
            if score < 0:
                ien[mask] = ien[mask][:, [0, 2, 1]]
                n_flipped += int(mask.sum())
            components.append({
                "id": cid,
                "n_tri": int(mask.sum()),
                "outward_score": score,
            })

        if not dry_run:
            self.ien = ien

        if faultmesher.verbose:
            verb = "would do" if dry_run else "did"
            print(f"fix_winding ({verb}): {int(needs_flip_stitch.sum())} "
                  f"stitching flips, {n_flipped} orientation flips "
                  f"across {n_components} component(s)")
        if ambiguous:
            warnings.warn(
                f"fix_winding: components {ambiguous} have average normal "
                f"nearly perpendicular to the reference direction; "
                f"orientation may be wrong.",
                RuntimeWarning, stacklevel=2,
            )

        return {
            "n_components": n_components,
            "n_inconsistent": int(needs_flip_stitch.sum()),
            "n_flipped": n_flipped,
            "ambiguous": ambiguous,
            "components": components,
        }

    def check_winding(self, reference=(1, 1, 1)):
        """Report winding diagnostics without modifying the mesh."""
        return self.fix_winding(reference=reference, dry_run=True)

    def _winding_components(self):
        edges = defaultdict(list)
        for ti, (a, b, c) in enumerate(self.ien):
            for u, v in ((a, b), (b, c), (c, a)):
                edges[(min(u, v), max(u, v))].append((ti, (u, v)))

        n = self.n_cells
        comp = -np.ones(n, dtype=np.int64)
        flip = np.zeros(n, dtype=bool)
        cid = 0
        for seed in range(n):
            if comp[seed] != -1:
                continue
            comp[seed] = cid
            stack = [seed]
            while stack:
                ti = stack.pop()
                a, b, c = self.ien[ti]
                fi = flip[ti]
                for u, v in ((a, b), (b, c), (c, a)):
                    key = (min(u, v), max(u, v))
                    for tj, (uj, vj) in edges[key]:
                        if tj == ti or comp[tj] != -1:
                            continue
                        comp[tj] = cid
                        flip[tj] = (not fi) if (uj, vj) == (u, v) else fi
                        stack.append(tj)
            cid += 1
        return comp, flip

    def merge_meshes(self, tolerance=None):
        """Weld vertices closer than `tolerance` and rewrite connectivity.

        Useful after meshing two faults that share a trace endpoint: each
        fault produces its own vertex at the shared point, and this welds
        them. Defaults to 1% of the shortest edge in the mesh.
        """
        if tolerance is None:
            tolerance = 0.01 * self.edge_lengths().min()

        n_before = len(self.xyz)
        pairs = cKDTree(self.xyz).query_pairs(tolerance, output_type="ndarray")
        if len(pairs) == 0:
            faultmesher.log(f"merge_meshes: no pairs within {tolerance:.3g}")
            return

        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                           shape=(n_before, n_before))
        _, labels = connected_components(graph, directed=False)
        _, keep = np.unique(labels, return_index=True)

        self.xyz = self.xyz[keep]
        self.ien = labels[self.ien]
        faultmesher.log(
            f"merge_meshes: {n_before} -> {len(self.xyz)} vertices "
            f"({n_before - len(self.xyz)} welded, tol={tolerance:.3g})"
        )

    # file I/O

    def to_pyvista(self, cell_data=None, point_data=None):
        """Build a pyvista UnstructuredGrid from this mesh.

        Tags, if present, attach as cell_data["tag"].
        """
        import pyvista as pv
        n = self.n_cells
        cells = np.hstack((3 * np.ones((n, 1), int), self.ien)).ravel()
        grid = pv.UnstructuredGrid(cells, [pv.CellType.TRIANGLE] * n, self.xyz)
        if self.tags is not None:
            grid.cell_data["tag"] = np.asarray(self.tags, dtype=str)
        for k, v in (cell_data or {}).items():
            grid.cell_data[k] = v
        for k, v in (point_data or {}).items():
            grid.point_data[k] = v
        return grid

    def write(self, path, cell_data=None, point_data=None):
        """Write the mesh to disk.

        .vtu goes through pyvista (carries tags and extra data); .msh,
        .vtk and .stl go through meshio (geometry only).
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".vtu":
            self.to_pyvista(cell_data, point_data).save(path, binary=True)
        else:
            import meshio
            fmt = {".msh": "gmsh", ".vtk": "vtk", ".stl": "stl"}.get(ext)
            if fmt is None:
                raise ValueError(f"unsupported extension {ext}")
            meshio.write(path, meshio.Mesh(self.xyz, {"triangle": self.ien}),
                         file_format=fmt)
        faultmesher.log(f"wrote {path}")

    @classmethod
    def read(cls, path):
        """Load a triangle mesh from disk.

        .vtu reads via pyvista (preserves tags); others via meshio.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".vtu":
            import pyvista as pv
            g = pv.read(path)
            cells = np.asarray(g.cells).reshape(-1, 4)
            if not np.all(cells[:, 0] == 3):
                raise ValueError(f"{path} contains non-triangle cells")
            tags = (np.asarray(g.cell_data["tag"], dtype=object)
                    if "tag" in g.cell_data else None)
            return cls(xyz=np.asarray(g.points), ien=cells[:, 1:], tags=tags)
        import meshio
        m = meshio.read(path)
        return cls(xyz=m.points, ien=m.cells_dict["triangle"])