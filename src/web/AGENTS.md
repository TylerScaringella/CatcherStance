# Web Agent Guide

The frontend is a vanilla DOM application. Preserve existing routes, API contracts, mobile navigation, keyboard behavior, visible focus, skip links, live regions, dialog semantics, and reduced-motion behavior.

Active run polling must patch existing progress and result nodes instead of rebuilding the root view. This protects search focus, dialogs, scroll position, and media playback. Keep selected game/run context visible when moving between Games, Runs, and Results.

Pitch dialogs must preserve compact review, full source recovery, pose overlay, and bottom-sheet behavior. Use text and semantic state markers in addition to color. Test desktop and mobile layouts, filters, review mutations, export readiness, and no-console-error behavior.
