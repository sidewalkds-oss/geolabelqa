"""geolabelqa - quality assurance for geospatial machine-learning training labels."""

from .auditor import LabelAuditor
from .geometry import GeometryValidator, Issue
from .spatial_cv import SpatialBlockKFold

__all__ = ["GeometryValidator", "Issue", "LabelAuditor", "SpatialBlockKFold"]
__version__ = "0.1.0"
