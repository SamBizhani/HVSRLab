"""The application's pages, in navigation order."""

from .analysis import AnalysisPage
from .batch import BatchPage
from .bedrock_page import BedrockPage
from .data import DataPage
from .maps import MapsPage
from .model import ModelPage
from .overview import OverviewPage
from .sites import SitesPage
from .views3d import Views3DPage

#: (title, class) in the order they appear in the sidebar.
PAGES = [
    ("Overview", OverviewPage),
    ("Sites", SitesPage),
    ("Data & Windows", DataPage),
    ("H/V Analysis", AnalysisPage),
    ("Maps & Profiles", MapsPage),
    ("3D Views", Views3DPage),
    ("Bedrock", BedrockPage),
    ("1D Model", ModelPage),
    ("Batch & Export", BatchPage),
]

__all__ = ["PAGES", "AnalysisPage", "BatchPage", "BedrockPage", "DataPage",
           "MapsPage", "ModelPage", "OverviewPage", "SitesPage", "Views3DPage"]
