import argparse
import os
import sys

import faultmesher
from faultmesher import FaultSystem
from faultmesher.gmsh_backend import ALGORITHMS


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="faultmesher",
        description="Surface mesher for boundary-element fault models.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="mesh a FaultSystem from a YAML file")
    run.add_argument("config", help="path to YAML config")
    run.add_argument(
        "-o", "--out",
        help="output path. If omitted, writes <config_stem>.<format> next "
             "to the YAML. If a directory, writes <config_stem>.<format> "
             "inside it. If a filename, uses that name (format inferred "
             "from extension if present).",
    )
    run.add_argument("-f", "--format", default="vtu",
                     choices=("vtu", "msh", "vtk", "stl"),
                     help="output mesh format when not implied by -o "
                          "(default: vtu)")
    run.add_argument("--algorithm", default="delaunay",
                     choices=sorted(ALGORITHMS),
                     help="gmsh 2D meshing algorithm")
    run.add_argument("--no-weld", action="store_true",
                     help="skip vertex welding between fault meshes")
    run.add_argument("--no-optimize", action="store_true",
                     help="skip the Netgen optimizer pass")
    run.add_argument("-q", "--quiet", action="store_true",
                     help="suppress progress output")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return run_cmd(args)


def run_cmd(args):
    if args.quiet:
        faultmesher.verbose = False

    if not os.path.isfile(args.config):
        sys.stderr.write(f"config not found: {args.config}\n")
        return 1
    try:
        sys_ = FaultSystem.from_yaml(args.config)
    except (ValueError, OSError) as e:
        sys.stderr.write(f"failed to load config: {e}\n")
        return 1

    out_path = resolve_out_path(args.config, args.out, args.format)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    surf = sys_.build_surface_mesh(
        algorithm=args.algorithm,
        optimize=not args.no_optimize,
        intersections="none" if args.no_weld else "weld",
    )
    surf.write(out_path)

    if not args.quiet:
        print(f"  wrote {out_path}: {surf.n_cells} tris, {surf.n_points} pts")
    return 0


def resolve_out_path(config, out, fmt):
    """Decide where to write the mesh.

    Rules:
    - no -o: <config_stem>.<fmt> next to the config
    - -o is a directory (or ends with /): <config_stem>.<fmt> inside it
    - -o has a recognised extension: used verbatim
    - -o has no extension: treat as directory
    """
    stem = os.path.splitext(os.path.basename(config))[0]
    if out is None:
        return os.path.join(os.path.dirname(config) or ".", f"{stem}.{fmt}")

    looks_like_dir = (
        out.endswith(os.sep)
        or os.path.isdir(out)
        or os.path.splitext(out)[1] == ""
    )
    if looks_like_dir:
        return os.path.join(out, f"{stem}.{fmt}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())