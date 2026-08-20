"""
tests/conftest.py — Pytest configuration and shared fixtures.

Stubs out heavy ML dependencies (torch, transformers, shap, numpy) so the
test suite runs without the 417MB model weights or GPU environment.
These stubs are installed into sys.modules BEFORE any application module is
imported, so they are picked up transparently by all test files.
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub: numpy
# We stub only the submodules/attributes actually referenced at import time.
# Tests that need real numpy import it locally after the conftest has run.
# ---------------------------------------------------------------------------

numpy_stub = MagicMock(name="numpy")
numpy_stub.ndarray = object   # used in type hints
numpy_stub.array = MagicMock(return_value=[])
numpy_stub.abs = MagicMock(side_effect=lambda x: x)
numpy_stub.argsort = MagicMock(return_value=[])
sys.modules["numpy"] = numpy_stub
sys.modules["numpy.core"] = MagicMock()
sys.modules["numpy.core.multiarray"] = MagicMock()

# ---------------------------------------------------------------------------
# Stub: torch
# ---------------------------------------------------------------------------

torch_stub = MagicMock(name="torch")
torch_stub.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
torch_stub.Tensor = MagicMock
torch_stub.float32 = "float32"
torch_stub.device = MagicMock(return_value=MagicMock())

# torch.cuda
cuda_stub = MagicMock()
cuda_stub.is_available = MagicMock(return_value=False)
torch_stub.cuda = cuda_stub

sys.modules["torch"] = torch_stub
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()

# ---------------------------------------------------------------------------
# Stub: transformers
# ---------------------------------------------------------------------------

transformers_stub = MagicMock(name="transformers")
transformers_stub.AutoModelForSequenceClassification = MagicMock()
transformers_stub.AutoTokenizer = MagicMock()
sys.modules["transformers"] = transformers_stub

# ---------------------------------------------------------------------------
# Stub: shap
# ---------------------------------------------------------------------------

shap_stub = MagicMock(name="shap")
shap_stub.Explainer = MagicMock()
sys.modules["shap"] = shap_stub
sys.modules["shap.maskers"] = MagicMock()
