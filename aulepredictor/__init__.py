"""Computational tool for the prediction of metal-binding sites in proteins using deep convolutional neural networks."""

# Add imports here
from .aulepredictor import *
from .models.models import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
