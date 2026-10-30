"""Experimental factorial research; the production/default FHS is unchanged."""

from .generators import CANDIDATES, FactorialModel, fit_factorial, shrunk_drift

__all__ = ["CANDIDATES", "FactorialModel", "fit_factorial", "shrunk_drift"]
