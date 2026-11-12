# Backend Agent Guide

Flask routes in this directory are the compatibility boundary for the web client and existing consumers. Preserve route names, response fields, status meanings, and additive-only schema changes unless a migration is explicitly planned.

Require the existing admin session for mutations. Never expose admin tokens, credentials, signed media URLs, or filesystem paths. Validate run, clip, season, and file identifiers with the existing path-containment helpers.

Use the established atomic JSON and run-lock utilities for run state, reviews, settings, and exports. Keep live runs distinct from read-only sample runs. Background jobs must remain resumable; polling and incremental updates must not be broken by a route change. Test both success and failure paths, including stale cache, missing media, partial results, and restart/reload behavior.
