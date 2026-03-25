"""Compatibility shim for the compute entrypoint.

New deployments should use `app.compute_main:app` for compute images and
`app.load_balancer_main:app` for load-balancer images. This module stays in
place so existing imports and tests that reference `app.main` keep working.
"""

from app.compute_main import *  # noqa: F401,F403
