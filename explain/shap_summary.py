"""
Shim module so tests can import explain.shap_summary
while real implementation lives in src/explain/shap_summary.py.
"""

from importlib import import_module as _import_module

_mod = _import_module("src.explain.shap_summary")
compute_top_features = _mod.compute_top_features
ShapSummary = getattr(_mod, "ShapSummary", None)
__all__ = ["compute_top_features", "ShapSummary"]
