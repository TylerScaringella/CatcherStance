# Research Agent Guide

Research scripts measure or prepare artifacts; they do not define production behavior. Generated reports and annotation manifests stay in ignored generated directories. Do not commit local audit output, videos, credentials, or exploratory notebooks.

Use canonical dataset paths and preserve raw labels. Group splits by source video or game so related clips cannot leak between train and test. Record dataset hashes, feature schema, model versions, timing policy, runtime, and class counts in local outputs. Rare classes, especially Squat, need explicit support warnings and must not be presented as reliable from tiny samples.

Active entry points include `audit_model.py`, `audit_abstentions.py`, `benchmark_staged_pipeline.py`, `replay_abstention_recovery.py`, and `prepare_annotation_manifest.py`. Historical summaries and notebook metrics are not current production claims. Update the relevant specification or README when a research conclusion changes a production contract.
