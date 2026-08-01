import { apiUrl, getRuns, getSchedule, startRun } from "./api.js";
import {
  badge,
  clear,
  element,
  formatDate,
  formatPercent,
  formatTime,
  statusLabel,
} from "./dom.js";

const ACTIVE = new Set(["queued", "running", "downloading", "detecting", "finalizing"]);
const state = {
  schedule: null,
  runs: [],
  selectedGameId: null,
  gameSearch: "",
  gameFilter: "all",
  resultSearch: "",
  stanceFilter: "all",
  qualityFilter: "all",
  connected: false,
  lastUpdated: null,
  pollFailures: 0,
  timer: null,
};

const content = document.querySelector("#appContent");
const activityText = document.querySelector("#activityText");
const activityButton = document.querySelector("#activityButton");
const connectionState = document.querySelector("#connectionState");
const liveRegion = document.querySelector("#liveRegion");
const toastRegion = document.querySelector("#toastRegion");
const pitchDialog = document.querySelector("#pitchDialog");
const pitchDialogContent = document.querySelector("#pitchDialogContent");

function route() {
  const parts = (location.hash || "#/games").slice(2).split("/").filter(Boolean);
  return { view: parts[0] || "games", id: parts[1] || null };
}

function activeRuns() {
  return state.runs.filter((run) => ACTIVE.has(run.status));
}

function latestRunForGame(gameId) {
  const candidates = state.runs.filter((run) => run.game?.id === gameId);
  const live = candidates.filter((run) => run.source === "live");
  return (live.length ? live : candidates)[0] || null;
}

function rerenderSearch(renderFunction, selector) {
  renderFunction();
  const input = document.querySelector(selector);
  if (input) {
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
}

function toast(message, variant = "info") {
  const item = element("div", { className: `toast toast-${variant}`, text: message });
  toastRegion.append(item);
  window.setTimeout(() => item.remove(), 4200);
}

function updateChrome() {
  const count = activeRuns().length;
  activityText.textContent = count ? `${count} active ${count === 1 ? "run" : "runs"}` : "No active runs";
  activityButton.classList.toggle("is-active", count > 0);
  connectionState.textContent = state.connected
    ? `Live · ${state.lastUpdated?.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) || "now"}`
    : "Reconnecting";
  connectionState.classList.toggle("is-offline", !state.connected);

  const current = route().view;
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.classList.toggle(
      "active",
      link.dataset.nav === current || (current === "results" && link.dataset.nav === "runs"),
    );
  });
}

function pageHeader(kicker, title, description, actions = []) {
  return element("header", { className: "page-header" }, [
    element("div", {}, [
      element("p", { className: "eyebrow", text: kicker }),
      element("h1", { text: title }),
      element("p", { className: "page-description", text: description }),
    ]),
    element("div", { className: "page-actions" }, actions),
  ]);
}

function progressBar(run) {
  const progress = run.progress || {};
  const percent = Number(progress.percent);
  const determinate = Number.isFinite(percent);
  return element("div", { className: "run-progress" }, [
    element("div", { className: "progress-copy" }, [
      element("span", { text: run.message || statusLabel(run.status) }),
      element("strong", { text: determinate ? `${percent.toFixed(0)}%` : "Working" }),
    ]),
    element("div", {
      className: `progress-track${determinate ? "" : " indeterminate"}`,
      role: "progressbar",
      "aria-valuemin": 0,
      "aria-valuemax": 100,
      "aria-valuenow": determinate ? percent : undefined,
    }, [element("span", { style: determinate ? `width:${Math.min(100, percent)}%` : "" })]),
  ]);
}

function runBadge(run) {
  const variant = run.status === "complete"
    ? "success"
    : run.status === "failed" || run.status === "interrupted"
      ? "danger"
      : ACTIVE.has(run.status)
        ? "progress"
        : "neutral";
  return badge(run.read_only ? "Sample" : statusLabel(run.status), run.read_only ? "sample" : variant);
}

function gameCard(game) {
  const run = latestRunForGame(game.id);
  const selected = state.selectedGameId === game.id;
  return element("button", {
    className: `schedule-card${selected ? " selected" : ""}`,
    type: "button",
    on: { click: () => { state.selectedGameId = game.id; render(); } },
  }, [
    element("span", { className: "schedule-date", text: formatDate(game.date).replace(", 2026", "") }),
    element("span", { className: "schedule-opponent", text: game.opponent }),
    element("span", { className: "schedule-meta", text: `${game.location} · ${game.result || "Scheduled"}` }),
    run ? runBadge(run) : badge("Not run"),
  ]);
}

function renderGames() {
  const games = state.schedule?.games || [];
  if (!state.selectedGameId && games.length) state.selectedGameId = games[0].id;
  const query = state.gameSearch.toLowerCase();
  const filtered = games.filter((game) => {
    const matchesText = `${game.opponent} ${game.location}`.toLowerCase().includes(query);
    const run = latestRunForGame(game.id);
    const matchesStatus =
      state.gameFilter === "all" ||
      (state.gameFilter === "complete" && run?.status === "complete") ||
      (state.gameFilter === "active" && run && ACTIVE.has(run.status)) ||
      (state.gameFilter === "not-run" && !run);
    return matchesText && matchesStatus;
  });
  const game = games.find((item) => item.id === state.selectedGameId) || filtered[0];
  const run = game ? latestRunForGame(game.id) : null;

  const search = element("input", {
    id: "gameSearch",
    type: "search",
    value: state.gameSearch,
    placeholder: "Search opponent or location",
    "aria-label": "Search games",
    on: {
      input: (event) => {
        state.gameSearch = event.target.value;
        rerenderSearch(renderGames, "#gameSearch");
      },
    },
  });
  const filter = element("select", {
    "aria-label": "Filter games",
    on: { change: (event) => { state.gameFilter = event.target.value; renderGames(); } },
  }, [
    ["all", "All games"], ["complete", "Completed"], ["active", "Active"], ["not-run", "Not run"],
  ].map(([value, text]) => element("option", { value, text, selected: state.gameFilter === value })));

  const section = element("section", { className: "page" });
  section.append(pageHeader(
    "2026 Duke Baseball",
    "Game workspace",
    "Start a run, follow its progress, and move directly into pitch-level review.",
  ));
  const layout = element("div", { className: "games-layout" });
  const rail = element("aside", { className: "schedule-panel", "aria-label": "Game schedule" }, [
    element("div", { className: "panel-toolbar" }, [search, filter]),
    element("div", { className: "schedule-list" },
      filtered.length ? filtered.map(gameCard) : [element("div", { className: "empty-state", text: "No games match these filters." })]),
  ]);

  const detail = element("article", { className: "game-detail" });
  if (!game) {
    detail.append(element("div", { className: "empty-state", text: "Select a game to continue." }));
  } else {
    const runAction = run?.status === "complete"
      ? element("a", { className: "button primary", href: `#/results/${run.id}`, text: "Review results" })
      : element("button", {
          className: "button primary",
          type: "button",
          disabled: run && ACTIVE.has(run.status),
          text: run && ACTIVE.has(run.status) ? "Run in progress" : run?.status === "ready" ? "Run detection" : "Start detection",
          on: { click: () => submitRun(game, detail) },
        });
    detail.append(
      element("div", { className: "game-hero" }, [
        element("div", {}, [
          element("p", { className: "eyebrow", text: game.conference ? "ACC matchup" : "Non-conference" }),
          element("h2", { text: game.opponent }),
          element("p", { className: "game-line", text: `${formatDate(game.date)} · ${game.location}` }),
        ]),
        runAction,
      ]),
      element("div", { className: "score-strip" }, [
        element("span", { text: game.result || "Scheduled" }),
        element("span", { text: run ? `${run.result_count || 0} pitch results` : "No analysis yet" }),
        run ? runBadge(run) : badge("Ready to start"),
      ]),
    );
    if (run && ACTIVE.has(run.status)) detail.append(progressBar(run));
    if (run?.status === "failed" || run?.status === "interrupted") {
      detail.append(element("div", { className: "callout danger" }, [
        element("strong", { text: statusLabel(run.status) }),
        element("span", { text: run.message || "This run needs attention." }),
      ]));
    }
    const advanced = element("details", { className: "advanced-settings" }, [
      element("summary", { text: "Run settings" }),
      element("div", { className: "settings-grid" }, [
        element("label", {}, [
          element("span", { text: "TruMedia game or pitch-card URL" }),
          element("input", {
            id: "trumediaUrl",
            type: "url",
            value: game.trumedia_url || "https://duke-ncaabaseball.trumedianetworks.com/baseball/",
          }),
        ]),
        element("label", { className: "check-row" }, [
          element("input", { id: "forceRedownload", type: "checkbox" }),
          element("span", { text: "Force a fresh download instead of resuming" }),
        ]),
      ]),
    ]);
    if (!run?.read_only) detail.append(advanced);
    if (run?.read_only) {
      detail.append(element("div", { className: "callout" }, [
        element("strong", { text: "Checked-in sample" }),
        element("span", { text: "This fixture is read-only and can be reviewed without TruMedia access." }),
      ]));
    }
  }
  layout.append(rail, detail);
  section.append(layout);
  clear(content).append(section);
}

async function submitRun(game, container) {
  const button = container.querySelector(".button.primary");
  const url = container.querySelector("#trumediaUrl")?.value;
  const force = container.querySelector("#forceRedownload")?.checked || false;
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    const run = await startRun({ game_id: game.id, trumedia_url: url, force_redownload: force });
    state.runs = [run, ...state.runs.filter((item) => item.id !== run.id)];
    liveRegion.textContent = `Run started for ${game.opponent}`;
    toast("Run started. You can continue using the application.", "success");
    location.hash = "#/runs";
    schedulePoll(500);
  } catch (error) {
    toast(error.message, "danger");
    button.disabled = false;
    button.textContent = "Try again";
  }
}

function renderRuns() {
  const runs = [...state.runs].sort((a, b) => {
    const activeDifference = Number(ACTIVE.has(b.status)) - Number(ACTIVE.has(a.status));
    return activeDifference || Number(b.updated_at || 0) - Number(a.updated_at || 0);
  });
  const section = element("section", { className: "page" });
  section.append(pageHeader(
    "Processing",
    "Run activity",
    "Jobs continue in the background while you review games or completed results.",
    [element("button", { className: "button ghost", type: "button", text: "Refresh", on: { click: () => refresh(true) } })],
  ));
  const grid = element("div", { className: "runs-grid" });
  if (!runs.length) {
    grid.append(element("div", { className: "empty-state large" }, [
      element("h2", { text: "No runs yet" }),
      element("p", { text: "Choose a game to begin a catcher stance analysis." }),
      element("a", { className: "button primary", href: "#/games", text: "Browse games" }),
    ]));
  }
  for (const run of runs) {
    const card = element("article", { className: `run-card${ACTIVE.has(run.status) ? " active-run" : ""}` }, [
      element("div", { className: "run-card-head" }, [
        element("div", {}, [
          element("p", { className: "eyebrow", text: formatDate(run.game?.date) }),
          element("h2", { text: run.game?.opponent || run.id }),
          element("p", { className: "muted", text: run.game?.location || run.id }),
        ]),
        runBadge(run),
      ]),
      ACTIVE.has(run.status)
        ? progressBar(run)
        : element("p", { className: "run-message", text: run.message || statusLabel(run.status) }),
      element("div", { className: "run-facts" }, [
        element("span", { text: `${run.result_count || 0} results` }),
        element("span", { text: `${run.manifest?.downloaded || 0}/${run.manifest?.total || 0} clips` }),
        element("span", { text: run.read_only ? "Read-only fixture" : "Live run" }),
      ]),
      run.status === "complete"
        ? element("a", { className: "button secondary", href: `#/results/${run.id}`, text: "Open results" })
        : element("a", { className: "button ghost", href: "#/games", text: "View game" }),
    ]);
    grid.append(card);
  }
  section.append(grid);
  clear(content).append(section);
}

function resultSummary(rows) {
  const accepted = rows.filter((row) => row.accepted);
  const count = (label) => rows.filter((row) => row.stance === label).length;
  const flagged = rows.filter((row) => row.quality_flags?.length || !row.accepted).length;
  const average = accepted.length
    ? accepted.reduce((sum, row) => sum + Number(row.confidence || 0), 0) / accepted.length
    : 0;
  return [
    ["Pitches", rows.length, "total"],
    ["LKD", count("LKD"), "left knee down"],
    ["RKD", count("RKD"), "right knee down"],
    ["Squat", count("Squat"), "traditional"],
    ["Needs attention", flagged, "flags or abstentions"],
    ["Avg. confidence", formatPercent(average), "accepted pitches"],
  ];
}

function resultRow(row, run) {
  const stance = row.stance || "Abstained";
  const quality = row.quality_flags?.length ? "Review" : row.accepted ? "Clear" : "Rejected";
  return element("button", {
    className: "result-row",
    type: "button",
    on: { click: () => openPitch(run, row) },
  }, [
    element("span", { className: "pitch-number", text: String(row.pitch_index ?? "—").padStart(2, "0") }),
    element("span", {}, [badge(stance, row.accepted ? "stance" : "danger")]),
    element("span", { className: "confidence-cell" }, [
      element("strong", { text: formatPercent(row.confidence, 1) }),
      element("span", { className: "confidence-track" }, [
        element("i", { style: `width:${Math.min(100, Number(row.confidence || 0) * 100)}%` }),
      ]),
    ]),
    element("span", { text: `${formatTime(row.window_start_seconds)}–${formatTime(row.window_end_seconds)}` }),
    element("span", {}, [badge(quality, quality === "Clear" ? "success" : quality === "Review" ? "warning" : "danger")]),
    element("span", { className: "row-action", text: "Review →" }),
  ]);
}

function renderResults(runId) {
  const run = state.runs.find((item) => item.id === runId);
  if (!run) {
    clear(content).append(element("section", { className: "page" }, [
      pageHeader("Results", "Run not found", "The requested run is unavailable."),
      element("a", { className: "button primary", href: "#/runs", text: "Back to runs" }),
    ]));
    return;
  }
  const query = state.resultSearch.toLowerCase();
  const rows = (run.results || []).filter((row) => {
    const textMatch = `${row.clip_id} ${row.stance} ${row.status}`.toLowerCase().includes(query);
    const stanceMatch = state.stanceFilter === "all" ||
      (state.stanceFilter === "abstained" ? !row.accepted : row.stance === state.stanceFilter);
    const qualityMatch = state.qualityFilter === "all" ||
      (state.qualityFilter === "flagged" && (row.quality_flags?.length || !row.accepted)) ||
      (state.qualityFilter === "clear" && row.accepted && !row.quality_flags?.length);
    return textMatch && stanceMatch && qualityMatch;
  });

  const section = element("section", { className: "page results-page" });
  section.append(pageHeader(
    run.read_only ? "Sample analysis" : "Completed analysis",
    `${run.game?.opponent || "Run"} results`,
    `${formatDate(run.game?.date)} · ${run.result_count} pitch-level classifications`,
    [
      element("a", { className: "button ghost", href: apiUrl(`/api/results/${run.id}/json`), text: "JSON", download: "" }),
      element("a", { className: "button secondary", href: apiUrl(`/api/results/${run.id}/csv`), text: "Export CSV", download: "" }),
    ],
  ));
  section.append(element("div", { className: "metrics-grid" },
    resultSummary(run.results || []).map(([label, value, note]) =>
      element("article", { className: "metric-card" }, [
        element("span", { text: label }),
        element("strong", { text: value }),
        element("small", { text: note }),
      ]),
    ),
  ));

  const search = element("input", {
    id: "resultSearch",
    type: "search",
    value: state.resultSearch,
    placeholder: "Search pitch or status",
    "aria-label": "Search results",
    on: {
      input: (event) => {
        state.resultSearch = event.target.value;
        rerenderSearch(() => renderResults(runId), "#resultSearch");
      },
    },
  });
  const stance = element("select", {
    "aria-label": "Filter by stance",
    on: { change: (event) => { state.stanceFilter = event.target.value; renderResults(runId); } },
  }, [["all", "All stances"], ["LKD", "LKD"], ["RKD", "RKD"], ["Squat", "Squat"], ["abstained", "Abstained"]]
    .map(([value, text]) => element("option", { value, text, selected: state.stanceFilter === value })));
  const quality = element("select", {
    "aria-label": "Filter by quality",
    on: { change: (event) => { state.qualityFilter = event.target.value; renderResults(runId); } },
  }, [["all", "All quality"], ["clear", "Clear"], ["flagged", "Needs attention"]]
    .map(([value, text]) => element("option", { value, text, selected: state.qualityFilter === value })));

  section.append(element("div", { className: "results-panel" }, [
    element("div", { className: "results-toolbar" }, [
      element("div", { className: "filter-group" }, [search, stance, quality]),
      element("span", { className: "muted", text: `${rows.length} of ${run.results?.length || 0} pitches` }),
    ]),
    element("div", { className: "result-table-head" }, [
      "Pitch", "Stance", "Confidence", "Set window", "Quality", "",
    ].map((text) => element("span", { text }))),
    element("div", { className: "result-list" },
      rows.length ? rows.map((row) => resultRow(row, run)) : [element("div", { className: "empty-state", text: "No pitches match these filters." })]),
  ]));
  clear(content).append(section);
}

function metadataItem(label, value) {
  return element("div", { className: "metadata-item" }, [
    element("span", { text: label }),
    element("strong", { text: value ?? "—" }),
  ]);
}

function openPitch(run, row) {
  const sourceUrl = apiUrl(`/api/runs/${encodeURIComponent(run.id)}/clips/${encodeURIComponent(row.clip_id)}/video`);
  const overlayUrl = apiUrl(`/api/runs/${encodeURIComponent(run.id)}/clips/${encodeURIComponent(row.clip_id)}/overlay.mjpg`);
  const video = element("video", { controls: "", preload: "metadata", src: sourceUrl });
  video.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(Number(row.window_start_seconds))) video.currentTime = Number(row.window_start_seconds);
  }, { once: true });
  const media = element("div", { className: "pitch-media" }, [video]);
  const sourceButton = element("button", {
    className: "segmented active",
    type: "button",
    text: "Source clip",
  });
  const overlayButton = element("button", {
    className: "segmented",
    type: "button",
    text: "Pose overlay",
    on: {
      click: () => {
        sourceButton.classList.remove("active");
        overlayButton.classList.add("active");
        media.replaceChildren(element("div", { className: "overlay-loading" }, [
          element("span", { className: "spinner", "aria-hidden": "true" }),
          element("span", { text: "Loading live pose overlay…" }),
          element("img", { src: overlayUrl, alt: "Catcher pose overlay" }),
        ]));
      },
    },
  });
  sourceButton.addEventListener("click", () => {
    overlayButton.classList.remove("active");
    sourceButton.classList.add("active");
    media.replaceChildren(video);
  });
  const close = element("button", {
    className: "icon-button",
    type: "button",
    "aria-label": "Close pitch details",
    text: "×",
    on: { click: () => pitchDialog.close() },
  });
  clear(pitchDialogContent).append(
    element("header", { className: "dialog-header" }, [
      element("div", {}, [
        element("p", { className: "eyebrow", text: `Pitch ${row.pitch_index}` }),
        element("h2", { id: "pitchDialogTitle", text: row.stance || "Abstained pitch" }),
        element("p", { className: "muted", text: row.clip_id }),
      ]),
      close,
    ]),
    element("div", { className: "media-mode" }, [sourceButton, overlayButton]),
    media,
    element("div", { className: "dialog-body" }, [
      element("div", { className: "metadata-grid" }, [
        metadataItem("Confidence", formatPercent(row.confidence, 1)),
        metadataItem("Camera quality", formatPercent(row.camera_quality, 0)),
        metadataItem("Impact", formatTime(row.impact_seconds)),
        metadataItem("Set window", `${formatTime(row.window_start_seconds)}–${formatTime(row.window_end_seconds)}`),
        metadataItem("Valid pose frames", row.valid_frame_count),
        metadataItem("Status", row.accepted ? "Accepted" : statusLabel(row.rejection_reason)),
      ]),
      element("section", { className: "detail-section" }, [
        element("h3", { text: "Temporal vote" }),
        element("div", { className: "vote-list" },
          Object.entries(row.vote_distribution || {}).map(([label, value]) =>
            element("div", {}, [
              element("span", { text: label }),
              element("span", { className: "vote-track" }, [element("i", { style: `width:${Number(value) * 100}%` })]),
              element("strong", { text: formatPercent(value) }),
            ]),
          ),
        ),
      ]),
      element("section", { className: "detail-section" }, [
        element("h3", { text: "Quality and provenance" }),
        element("div", { className: "tag-list" }, [
          ...(row.quality_flags?.length ? row.quality_flags.map((flag) => badge(statusLabel(flag), "warning")) : [badge("No quality flags", "success")]),
          ...(row.detector_provenance || []).map((item) => badge(statusLabel(item), "neutral")),
        ]),
      ]),
    ]),
  );
  pitchDialog.showModal();
}

function render() {
  updateChrome();
  const current = route();
  if (current.view === "runs") renderRuns();
  else if (current.view === "results") renderResults(current.id);
  else renderGames();
}

async function refresh(announce = false) {
  try {
    const [schedule, runsPayload] = await Promise.all([getSchedule(), getRuns()]);
    state.schedule = schedule;
    state.runs = runsPayload.runs || [];
    state.connected = true;
    state.lastUpdated = new Date();
    state.pollFailures = 0;
    if (announce) toast("Run status refreshed.", "success");
    render();
  } catch (error) {
    state.connected = false;
    state.pollFailures += 1;
    updateChrome();
    if (announce) toast(error.message, "danger");
  } finally {
    schedulePoll();
  }
}

function schedulePoll(delay) {
  window.clearTimeout(state.timer);
  const activeDelay = activeRuns().length ? 2500 : 10000;
  const backoff = Math.min(30000, activeDelay * 2 ** state.pollFailures);
  state.timer = window.setTimeout(() => refresh(false), delay ?? backoff);
}

window.addEventListener("hashchange", render);
window.addEventListener("focus", () => refresh(false));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") refresh(false);
});
activityButton.addEventListener("click", () => { location.hash = "#/runs"; });
pitchDialog.addEventListener("close", () => clear(pitchDialogContent));
pitchDialog.addEventListener("click", (event) => {
  if (event.target === pitchDialog) pitchDialog.close();
});

refresh(false);
