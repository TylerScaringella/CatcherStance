# Backend

Runs the Flask API used by the web interface.

Main entry point: `src/app.py`

Important modules:
- `routes.py`: API routes and static file serving.
- `jobs.py`: background job state, result loading, and run resumption.
- `schedule.py`: Duke schedule loading and refresh logic.
- `config.py`: backend paths and default URLs.
