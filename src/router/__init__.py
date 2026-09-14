"""Adaptive router: feature building, training and analysis scripts.

`build_router` is re-exported so the historical call style stays valid:
    from src.router import build_router
"""
from .router import build_router

__all__ = ["build_router"]
