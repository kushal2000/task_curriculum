"""Evaluation harnesses.

Kept inside the package rather than under ``experiments/`` so that every interpreter that can
import ``isaacsimenvs`` runs the *same* code object for a measurement. A protocol difference
between two runs then cannot come from two copies of the harness drifting apart.
"""
