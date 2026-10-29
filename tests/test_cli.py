import json

from geolabelqa.cli import main


def test_validate_reports_errors(tmp_path, capsys):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"label": "forest"},
             "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}},
            {"type": "Feature", "properties": {"label": "water"},
             "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}},
        ],
    }
    path = tmp_path / "a.geojson"
    path.write_text(json.dumps(fc))
    assert main(["validate", str(path)]) == 1
    assert "self_intersection" in capsys.readouterr().out


def test_demo_then_audit(tmp_path, capsys):
    csv = tmp_path / "demo.csv"
    out = tmp_path / "report.csv"
    assert main(["demo", "-o", str(csv)]) == 0
    feats = "blue,green,red,nir,swir1,swir2,ndvi,ndwi,ndbi"
    assert main(["audit", str(csv), "--x", "x", "--y", "y", "--label", "label", "--features", feats, "-o", str(out)]) == 0
    assert out.exists()
    assert "flagged by confident learning" in capsys.readouterr().out
