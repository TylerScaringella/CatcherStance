# Stance Pipeline Agent Guide

The analyzer stages are ordered intentionally: scene/camera handling, catcher proposal and identity selection, event anchoring, bounded pre-delivery window selection, temporal feature extraction, classification, and quality/abstention reporting. Do not classify a post-receive or transition frame as the set stance.

Catcher identity is a safety gate. Preserve temporal continuity, reject candidate swaps, and retain diagnostics when no candidate is selected. Temporary detector misses may only be recovered when continuity and geometry support the same identity. Missing timing evidence, weak coverage, and ambiguous identity remain abstentions.

The feature schema and classifier artifact must stay compatible with `models/classifier/metadata.json`. Any model or threshold change needs grouped evaluation, rare-class reporting, timing regression, runtime measurement, and a checked-in test or metadata update. Never use a historical row-wise metric as a current quality claim.
