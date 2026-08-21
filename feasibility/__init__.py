"""Feasibility checking and the objective function.

``collision_detector`` is the exact, object-based geometry used by the original
study: ray-casting point-in-polygon, segment intersection and a polygon-pair
overlap test.  ``evaluator`` is the vectorised re-implementation the experiments
actually call; it folds the same feasibility tests and the travel-distance
objective into one NumPy pass.  ``tests.test_fast_eval_equivalence``
asserts the two agree exactly.
"""
