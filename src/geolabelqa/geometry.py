"""Rule-based quality checks for annotated polygons (GeoJSON).

Before a label's *class* can be questioned, its *geometry* has to be sound.
These checks cover the defects that most often break training-data pipelines
built from hand-digitised polygons. They are implemented without GEOS so the
package stays light; for production-scale validation use Shapely/PostGIS.

Rules
-----
invalid_coordinate     NaN/inf coordinates, or lon/lat outside WGS84 bounds
unclosed_ring          first and last vertex differ
too_few_vertices       ring has fewer than 4 positions (3 distinct + closing)
repeated_vertex        consecutive duplicate vertices
self_intersection      two non-adjacent ring edges cross
zero_area              ring encloses no area
sliver                 Polsby-Popper compactness below a threshold
winding_order          exterior not counter-clockwise / holes not clockwise (RFC 7946)
hole_outside_shell     an interior ring vertex lies outside the exterior ring
duplicate_geometry     geometry identical to an earlier feature's
missing_label          label property absent or empty
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

Ring = list[tuple[float, float]]


@dataclass(frozen=True)
class Issue:
    feature: int
    rule: str
    severity: str  # "error" breaks downstream processing, "warning" needs review
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def signed_area(ring: Ring) -> float:
    """Shoelace area; positive for counter-clockwise rings."""
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:]))


def perimeter(ring: Ring) -> float:
    return sum(math.dist(a, b) for a, b in zip(ring, ring[1:]))


def polsby_popper(ring: Ring) -> float:
    """Compactness 4*pi*A/P^2: 1 for a circle, near 0 for slivers."""
    p = perimeter(ring)
    return 4 * math.pi * abs(signed_area(ring)) / (p * p) if p else 0.0


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(p1, p2, q1, q2) -> bool:
    """True if closed segments p1-p2 and q1-q2 share at least one point."""
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4):
        return True
    return (
        (d1 == 0 and _on_segment(q1, q2, p1))
        or (d2 == 0 and _on_segment(q1, q2, p2))
        or (d3 == 0 and _on_segment(p1, p2, q1))
        or (d4 == 0 and _on_segment(p1, p2, q2))
    )


def self_intersects(ring: Ring) -> bool:
    """Check a closed ring for crossing non-adjacent edges (O(n^2))."""
    edges = list(zip(ring, ring[1:]))
    n = len(edges)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:  # first and last edges share the closing vertex
                continue
            if segments_intersect(*edges[i], *edges[j]):
                return True
    return False


def point_in_ring(pt, ring: Ring) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = pt
    inside = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _polygons(geometry: dict) -> list[list[Ring]]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "Polygon":
        polys = [coords]
    elif kind == "MultiPolygon":
        polys = coords
    else:
        return []
    return [[[tuple(map(float, pt[:2])) for pt in ring] for ring in poly] for poly in polys]


class GeometryValidator:
    """Validate the polygons of a GeoJSON FeatureCollection.

    Parameters
    ----------
    label_property:
        Name of the property that holds the class label.
    geographic:
        Treat coordinates as WGS84 lon/lat and check their bounds.
    sliver_threshold:
        Polsby-Popper score under which a ring is reported as a sliver.
    """

    def __init__(self, label_property: str = "label", geographic: bool = True, sliver_threshold: float = 0.05):
        self.label_property = label_property
        self.geographic = geographic
        self.sliver_threshold = sliver_threshold

    def validate(self, collection: dict) -> list[Issue]:
        issues: list[Issue] = []
        seen: dict[str, int] = {}
        for i, feature in enumerate(collection.get("features", [])):
            issues += self.validate_feature(i, feature)
            key = json.dumps(feature.get("geometry"), sort_keys=True)
            if key in seen:
                issues.append(Issue(i, "duplicate_geometry", "warning", f"same geometry as feature {seen[key]}"))
            else:
                seen[key] = i
        return issues

    def validate_file(self, path: str | Path) -> list[Issue]:
        with open(path, encoding="utf-8") as f:
            return self.validate(json.load(f))

    def validate_feature(self, i: int, feature: dict) -> list[Issue]:
        issues: list[Issue] = []
        label = (feature.get("properties") or {}).get(self.label_property)
        if label is None or str(label).strip() == "":
            issues.append(Issue(i, "missing_label", "error", f"property '{self.label_property}' is missing or empty"))

        geometry = feature.get("geometry") or {}
        polygons = _polygons(geometry)
        if not polygons:
            issues.append(Issue(i, "unsupported_geometry", "error", f"expected Polygon/MultiPolygon, got {geometry.get('type')}"))
            return issues

        for poly in polygons:
            for r, ring in enumerate(poly):
                issues += self._check_ring(i, ring, exterior=(r == 0))
            if poly and len(poly) > 1 and len(poly[0]) >= 4:
                for hole in poly[1:]:
                    if any(not point_in_ring(pt, poly[0]) for pt in hole[:-1]):
                        issues.append(Issue(i, "hole_outside_shell", "error", "interior ring extends outside the exterior ring"))
        return issues

    def _check_ring(self, i: int, ring: Ring, exterior: bool) -> list[Issue]:
        which = "exterior" if exterior else "interior"
        bad = [
            pt for pt in ring
            if not all(math.isfinite(c) for c in pt)
            or (self.geographic and not (-180 <= pt[0] <= 180 and -90 <= pt[1] <= 90))
        ]
        if bad:
            return [Issue(i, "invalid_coordinate", "error", f"{which} ring has invalid coordinate {bad[0]}")]

        issues = []
        if ring and ring[0] != ring[-1]:
            issues.append(Issue(i, "unclosed_ring", "error", f"{which} ring is not closed"))
            ring = ring + [ring[0]]
        if len(ring) < 4:
            return issues + [Issue(i, "too_few_vertices", "error", f"{which} ring has {len(ring)} positions, need >= 4")]

        if any(a == b for a, b in zip(ring, ring[1:])):
            issues.append(Issue(i, "repeated_vertex", "warning", f"{which} ring has consecutive duplicate vertices"))
            ring = [pt for k, pt in enumerate(ring) if k == 0 or pt != ring[k - 1]]
            if len(ring) < 4:
                return issues + [Issue(i, "too_few_vertices", "error", f"{which} ring collapses after removing duplicates")]

        # Self-intersection first: a bow-tie's lobes cancel out to zero signed area.
        if self_intersects(ring):
            return issues + [Issue(i, "self_intersection", "error", f"{which} ring crosses itself")]
        area = signed_area(ring)
        if area == 0:
            return issues + [Issue(i, "zero_area", "error", f"{which} ring has zero area")]
        if (pp := polsby_popper(ring)) < self.sliver_threshold:
            issues.append(Issue(i, "sliver", "warning", f"{which} ring compactness {pp:.3f} < {self.sliver_threshold}"))
        if (area > 0) != exterior:
            want = "counter-clockwise" if exterior else "clockwise"
            issues.append(Issue(i, "winding_order", "warning", f"{which} ring should be {want} (RFC 7946)"))
        return issues
