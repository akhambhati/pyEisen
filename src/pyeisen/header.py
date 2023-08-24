"""header.py
Information about pyeisen.
"""

from importlib.metadata import metadata

# Header #
__package_name__ = __name__
_metadata = metadata(__package_name__)


__author__ = _metadata['Author']
__email__ = _metadata['Author-email']

__license__ = _metadata['License']

__version__ = _metadata['Version']
__status__ = _metadata['Classifier']

__all__ = [
    "__package_name__",
    "__author__",
    "__email__",
    "__license__",
    "__version__",
    "__status__",
]
