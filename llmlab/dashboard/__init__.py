"""The dashboard: a local web app over the lab's database.

api.py turns the database and experiment files into JSON, server.py serves that JSON
and the page in static/, and starts and stops runs. Nothing here measures anything:
runs are ordinary `llmlab run` processes, and the dashboard watches their live log.
"""
