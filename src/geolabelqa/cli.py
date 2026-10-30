"""Command-line interface.

    geolabelqa validate annotations.geojson [--label-property class] [--projected]
    geolabelqa audit samples.csv --x x --y y --label label [--features b1,b2,...] [-o report.csv]
    geolabelqa demo [-o demo.csv]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import pandas as pd

from .auditor import LabelAuditor
from .geometry import GeometryValidator
from .synthetic import CLASSES, inject_noise, make_landcover


def _validate(args) -> int:
    validator = GeometryValidator(args.label_property, geographic=not args.projected, sliver_threshold=args.sliver)
    issues = validator.validate_file(args.path)
    for issue in issues:
        print(f"feature {issue.feature:>5}  {issue.severity:<7}  {issue.rule:<20} {issue.message}")
    counts = Counter(i.severity for i in issues)
    print(f"\n{counts['error']} error(s), {counts['warning']} warning(s)")
    return 1 if counts["error"] else 0


def _audit(args) -> int:
    df = pd.read_csv(args.path)
    reserved = {args.x, args.y, args.label}
    features = args.features.split(",") if args.features else [
        c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
    ]
    auditor = LabelAuditor(block_size=args.block_size, k=args.k, cv=args.cv)
    report = auditor.audit(df[features].to_numpy(), df[args.label].to_numpy(), df[[args.x, args.y]].to_numpy())
    if args.output:
        report.to_csv(args.output, index=False)
        print(f"wrote {len(report)} rows to {args.output}")
    print(f"{int(report['cl_flag'].sum())} of {len(report)} samples flagged by confident learning")
    print(report.head(args.top).to_string(index=False, float_format="%.3f"))
    return 0


def _demo(args) -> int:
    ds = make_landcover(seed=args.seed)
    noisy, is_noisy = inject_noise(ds.y, ds.coords, "clustered", 0.15, seed=args.seed)
    df = pd.DataFrame(ds.X, columns=ds.feature_names)
    df.insert(0, "y", ds.coords[:, 1])
    df.insert(0, "x", ds.coords[:, 0])
    df["label"] = [CLASSES[c] for c in noisy]
    df["true_label"] = [CLASSES[c] for c in ds.y]
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} samples ({is_noisy.sum()} with corrupted labels) to {args.output}")
    print(f"try: geolabelqa audit {args.output} --x x --y y --label label --features {','.join(ds.feature_names)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geolabelqa", description="Quality assurance for geospatial ML training labels.")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="check annotation polygons in a GeoJSON file")
    v.add_argument("path")
    v.add_argument("--label-property", default="label")
    v.add_argument("--projected", action="store_true", help="skip WGS84 lon/lat bounds checks")
    v.add_argument("--sliver", type=float, default=0.05, help="Polsby-Popper sliver threshold")
    v.set_defaults(func=_validate)

    a = sub.add_parser("audit", help="rank labelled point samples by label-error suspicion")
    a.add_argument("path", help="CSV with coordinates, label and numeric features")
    a.add_argument("--x", required=True, help="x / easting column (projected units)")
    a.add_argument("--y", required=True, help="y / northing column (projected units)")
    a.add_argument("--label", required=True)
    a.add_argument("--features", help="comma-separated feature columns (default: all other numeric columns)")
    a.add_argument("--cv", choices=("random", "spatial"), default="random", help="fold design (see the paper)")
    a.add_argument("--block-size", type=float, default=10.0, help="spatial block side, used with --cv spatial")
    a.add_argument("--k", type=int, default=10)
    a.add_argument("--top", type=int, default=15)
    a.add_argument("-o", "--output")
    a.set_defaults(func=_audit)

    m = sub.add_parser("demo", help="write a synthetic noisy dataset to try the audit on")
    m.add_argument("-o", "--output", default="demo_samples.csv")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
