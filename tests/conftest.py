"""Global test configuration.

Windows CPU runs of the full suite intermittently die with a native access
violation inside MHA softmax (observed 2026-06-11; isolated files always pass).
Mitigations, all required together:
- single-threaded torch (multi-threaded OpenMP makes the crash far more likely)
- KMP_DUPLICATE_LIB_OK guard (duplicate libiomp5md.dll from MKL + torch)
- gc between tests (the crash correlates with accumulated allocator pressure)
The env vars must be set before torch initializes its OpenMP runtime, hence
the import order here.
"""

import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # interop pool already started (e.g. torch imported by a plugin first);
    # intra-op + env pinning above still applies.
    pass


@pytest.fixture(autouse=True)
def _collect_garbage():
    yield
    gc.collect()
