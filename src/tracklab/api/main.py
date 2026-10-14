"""Uvicorn entry point for the TrackLab API."""

from tracklab.api.application import create_app

app = create_app()
