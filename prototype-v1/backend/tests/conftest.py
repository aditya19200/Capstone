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
# numpy is NOT stubbed: pandas (routers/batches.py) needs a real numpy at
# import time, and shap_service.py does real array math (np.abs/np.argsort)
# that a MagicMock stub can't reproduce correctly. numpy itself is a small,
# pure-C-extension dependency with prebuilt wheels — unlike torch/transformers
# below, there's no weights/GPU cost to installing the real thing.
# ---------------------------------------------------------------------------

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
