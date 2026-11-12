# Backend

Runs the Flask API used by the web interface.

Main entry point: `src/app.py`

Important modules:
- `routes.py`: API routes and static file serving.
- `jobs.py`: background job state, result loading, and run resumption.
- `storage.py`: allowlisted live/example run resolution, atomic writes, temporary
  lifecycle, identifier validation, and safe manifest media lookup.

The backend never accepts a client-provided filesystem path. Runtime runs are
writable, checked-in examples are read-only, and schedule refreshes are written
to the ignored runtime cache.
- `schedule.py`: Duke schedule loading, season settings, validation, and admin-triggered refresh logic.
- `config.py`: backend paths and default URLs.

Schedule refreshes are intentionally manual. The active season and refresh metadata are stored in
the ignored runtime cache at `data/cache/schedule_settings.json`; the UI exposes the management
workflow after the existing admin token is unlocked.
