# Copyright (c) 2026
"""Path bootstrap for the side-mounted multi-metric callback tests.

These tests intentionally live outside ``tests/tests_pytorch`` so the upstream
Lightning test tree is untouched.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
