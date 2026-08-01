# Codex attribution: OpenAI Codex generated this Flask app factory module as
# part of the backend API scaffolding, with project-specific review and edits.
import os

from flask import Flask

from .routes import register_routes
from .schedule import start_schedule_refresh_job
from .storage import initialize_storage


def create_app():
    initialize_storage()
    app = Flask(__name__, static_folder=None)
    app.secret_key = os.environ.get("CATCHER_STANCE_SECRET_KEY") or os.urandom(32)
    app.config.update(
        MAX_CONTENT_LENGTH=1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=os.environ.get("CATCHER_STANCE_SECURE_COOKIES") == "1",
    )
    register_routes(app)
    return app


def main():
    start_schedule_refresh_job()
    app = create_app()
    print("Catcher Stance web app running at http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, threaded=True)
