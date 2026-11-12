# Agent Guide

## Purpose

This repository is a Duke baseball operations tool for turning pitch clips into reviewable catcher-stance results. Inspect existing behavior and contracts before changing them. Prefer a small, test-backed change over a broad rewrite.

## Source of truth

- Production behavior lives in `src/`; research scripts do not define runtime behavior.
- API routes, stored result fields, exports, and sample fixtures are compatibility contracts.
- `models/classifier/metadata.json` describes the checked-in classifier artifact.
- Generated reports, annotation manifests, runtime state, credentials, signed URLs, and video stay local and ignored.
- Read the nearest nested `AGENTS.md` before editing backend, pipeline, web, or research code.

## Map and safe commands

- `src/backend/`: Flask routes, storage, jobs, schedule, review, and media recovery.
- `src/stance_pipeline/`: staged catcher/event/pose/classifier pipeline.
- `src/web/`: vanilla DOM application.
- `tests/`: unit, contract, and optional browser tests.
- `research/`: reproducible audits, benchmarks, annotations, and specifications.
- `data/examples/`: checked-in sample run; other `data/` content is normally private or generated.

Run `PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v`. With Chromium, run the browser command in the root README. Use `git diff --check`, compile/import checks, and a local app smoke check before handoff.

## Engineering rules

Keep API payload shapes and existing URLs backward compatible. Preserve atomic writes, path containment, run locks, resumability, polling, accessibility, and read-only sample behavior. Never log or commit credentials, media URLs, or private runtime data. Research outputs belong under ignored generated directories. Do not change model thresholds or event timing policy without grouped evidence and regression tests.

Use a `codex/` feature branch, keep the work scoped, update documentation and tests with behavior changes, and create a draft PR with actual validation results. Do not put transient task status in durable guides.
