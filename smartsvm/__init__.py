__version__ = "1.0.3"

# shortcut
from .smartsvm import SmartSVM
from .error_estimate import (
    hp_estimate,
    hp_binary,
    compute_ovr_error,
    compute_error_graph,
)
