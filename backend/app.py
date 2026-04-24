# Codex attribution: OpenAI Codex generated this Flask app factory module as
# part of the backend API scaffolding, with project-specific review and edits.
from flask import Flask

from .routes import register_routes
from .schedule import start_schedule_refresh_job


def create_app():
    app = Flask(__name__, static_folder=None)
    register_routes(app)
    return app


def main():
    start_schedule_refresh_job()
    app = create_app()
    print("Catcher Stance web app running at http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, threaded=True)
