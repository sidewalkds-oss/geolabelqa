import math

import pytest

from geolabelqa.geometry import (
    GeometryValidator,
    point_in_ring,
    polsby_popper,
    segments_intersect,
    self_intersects,
    signed_area,
)

SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]  # counter-clockwise


def feature(rings, label="forest", kind="Polygon"):
    return {"type": "Feature", "properties": {"label": label}, "geometry": {"type": kind, "coordinates": rings}}


def rules(*features, **kwargs):
    issues = GeometryValidator(**kwargs).validate({"type": "FeatureCollection", "features": list(features)})
    return {i.rule for i in issues}


def test_signed_area_and_orientation():
    assert signed_area(SQUARE) == pytest.approx(1.0)
    assert signed_area(SQUARE[::-1]) == pytest.approx(-1.0)


def test_polsby_popper_square():
    assert polsby_popper(SQUARE) == pytest.approx(math.pi / 4)


def test_segments_intersect_cases():
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))
    assert segments_intersect((0, 0), (2, 0), (1, 0), (3, 0))  # collinear overlap


def test_bowtie_self_intersects():
    assert self_intersects([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    assert not self_intersects(SQUARE)


def test_point_in_ring():
    assert point_in_ring((0.5, 0.5), SQUARE)
    assert not point_in_ring((1.5, 0.5), SQUARE)


def test_valid_polygon_has_no_issues():
    assert rules(feature([SQUARE])) == set()


def test_detects_each_rule():
    assert "unclosed_ring" in rules(feature([SQUARE[:-1]]))
    assert "too_few_vertices" in rules(feature([[(0, 0), (1, 0), (0, 0)]]))
    assert "repeated_vertex" in rules(feature([[(0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]))
    assert "self_intersection" in rules(feature([[(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]]))
    assert "zero_area" in rules(feature([[(0, 0), (1, 0), (2, 0), (0, 0)]]))
    assert "sliver" in rules(feature([[(0, 0), (1, 0), (1, 0.001), (0, 0.001), (0, 0)]]))
    assert "winding_order" in rules(feature([SQUARE[::-1]]))
    assert "invalid_coordinate" in rules(feature([[(0, 0), (200, 0), (200, 1), (0, 1), (0, 0)]]))
    assert "missing_label" in rules(feature([SQUARE], label=""))
    assert "unsupported_geometry" in rules({"type": "Feature", "properties": {"label": "x"},
                                            "geometry": {"type": "Point", "coordinates": [0, 0]}})


def test_projected_mode_skips_lonlat_bounds():
    big = [(0, 0), (500, 0), (500, 500), (0, 500), (0, 0)]
    assert rules(feature([big]), geographic=False) == set()


def test_holes():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    good_hole = [(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]  # clockwise
    bad_hole = [(8, 8), (8, 12), (12, 12), (12, 8), (8, 8)]
    assert rules(feature([outer, good_hole])) == set()
    assert "hole_outside_shell" in rules(feature([outer, bad_hole]))


def test_duplicate_geometry_and_multipolygon():
    assert "duplicate_geometry" in rules(feature([SQUARE]), feature([SQUARE], label="water"))
    assert rules(feature([[SQUARE]], kind="MultiPolygon")) == set()
