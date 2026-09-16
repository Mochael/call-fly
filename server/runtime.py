"""Explicit backend selection; never silently swap calibrated model revisions."""
import os
if os.environ.get('MOSHI_BACKEND','mlx')=='torch':
    from .moshi_torch_engine import MoshiEngine, ROOT, MODEL, REVISION, RATE, FRAME, MAX_STEPS, ADAPTER
else:
    from .moshi_engine import MoshiEngine, ROOT, MODEL, REVISION, RATE, FRAME, MAX_STEPS
    from .model_reservoir import DEFAULT_ADAPTER as ADAPTER
