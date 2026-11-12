# Codex attribution: OpenAI Codex generated this Flask app factory module as
# part of the backend API scaffolding, with project-specific review and edits.
import os

from flask import Flask, jsonify

from .routes import register_routes
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

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    return app


def main():
    app = create_app()
    host = os.environ.get("CATCHER_STANCE_HOST", "127.0.0.1")
    port = int(os.environ.get("CATCHER_STANCE_PORT", "8000"))
    print(f"Catcher Stance web app running at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
