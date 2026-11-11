"""Answer-quality scoring: is a faster setting still as good?

A suite is a fixed set of tasks with known answers (suite.py). Each answer is graded by
code (scorers.py), and generated programs run against hidden tests in a guarded
subprocess (sandbox.py). No model grades another model or itself. run.py asks an engine
the tasks, and summary.py turns the graded answers into a score with an uncertainty.
"""
