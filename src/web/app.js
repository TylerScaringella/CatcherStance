import {
  apiUrl,
  confirmTruMediaMatch,
  getRun,
  getRunSummaries,
  getSchedule,
  getSeasons,
  getTruMediaStatus,
  reprocessRun,
  startRun,
  unlockAdmin,
  uploadTruMediaSession,
  validateTruMediaSession,
} from "./api.js";
import {
  badge,
  clear,
  element,
  formatDate,
  formatPercent,
  formatTime,
  statusLabel,
} from "./dom.js";

const ACTIVE = new Set([
  "queued", "running", "resolving_game", "discovering_pitches", "downloading",
  "detecting", "building_review", "cleaning_up", "finalizing",
]);
const PIPELINE_STAGES = ["discovering_pitches", "downloading", "detecting", "building_review"];
const state = {
  schedule: null,
  runs: [],
  selectedGameId: null,
  gameSearch: "",
  gameFilter: "all",
  siteFilter: "all",
  resultFilter: "all",
  conferenceFilter: "all",
  dateFrom: "",
  dateTo: "",
  season: 2026,
  seasons: [2026],
  trumedia: { status: "missing", connected: false },
  resultSearch: "",
  stanceFilter: "all",
  qualityFilter: "all",
  connected: false,
  lastUpdated: null,
  pollFailures: 0,
  timer: null,
  pollController: null,
  runsEtag: null,
  runDetails: new Map(),
};

const content = document.querySelector("#appContent");
const activityText = document.querySelector("#activityText");
const activityButton = document.querySelector("#activityButton");
const trumediaButton = document.querySelector("#trumediaButton");
const trumediaText = document.querySelector("#trumediaText");
const connectionState = document.querySelector("#connectionState");
const liveRegion = document.querySelector("#liveRegion");
const toastRegion = document.querySelector("#toastRegion");
const pitchDialog = document.querySelector("#pitchDialog");
const pitchDialogContent = document.querySelector("#pitchDialogContent");
const workflowDialog = document.querySelector("#workflowDialog");
const workflowDialogContent = document.querySelector("#workflowDialogContent");

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

function closeWorkflow() {
  if (workflowDialog.open) workflowDialog.close();
}

function workflowHeader(kicker, title, description) {
  return element("header", { className: "dialog-header" }, [
    element("div", {}, [
      element("p", { className: "eyebrow", text: kicker }),
      element("h2", { id: "workflowDialogTitle", text: title }),
      element("p", { className: "muted", text: description }),
    ]),
    element("button", {
      className: "icon-button", type: "button", "aria-label": "Close dialog", text: "×",
      on: { click: closeWorkflow },
    }),
  ]);
}

function openAuthentication(game = null, container = null, reprocess = false) {
  const token = element("input", { type: "password", autocomplete: "current-password", placeholder: "Admin token", "aria-label": "Admin token" });
  const file = element("input", { type: "file", accept: "application/json,.json", "aria-label": "Playwright storage-state JSON" });
  const status = element("p", {
    className: "form-status",
    text: state.trumedia.connected
      ? "The installed session is ready. Revalidate it or upload a replacement."
      : "Unlock this protected action, then upload an exported Playwright session.",
  });
  const submit = element("button", {
    className: "button primary", type: "button", text: state.trumedia.connected ? "Replace session" : "Upload session",
    on: { click: async () => {
      if (!token.value || !file.files?.[0]) {
        status.textContent = "An admin token and storage-state JSON file are required.";
        return;
      }
      submit.disabled = true;
      submit.textContent = "Validating headless session…";
      status.textContent = "The session is being checked against TruMedia. Credentials never enter the schedule or run data.";
      try {
        await unlockAdmin(token.value);
        state.trumedia = await uploadTruMediaSession(file.files[0]);
        toast("TruMedia session connected.", "success");
        updateChrome();
        closeWorkflow();
        if (game && container) await submitRun(game, container, reprocess);
      } catch (error) {
        status.textContent = error.message;
        submit.disabled = false;
        submit.textContent = "Try again";
      }
    } },
  });
  const revalidate = element("button", {
    className: "button secondary", type: "button", text: "Revalidate current",
    on: { click: async () => {
      if (!token.value) {
        status.textContent = "Enter the admin token before revalidating.";
        return;
      }
      revalidate.disabled = true;
      revalidate.textContent = "Validating…";
      status.textContent = "Checking the installed session against TruMedia.";
      try {
        await unlockAdmin(token.value);
        state.trumedia = await validateTruMediaSession();
        status.textContent = "The installed TruMedia session is valid.";
        toast("TruMedia session validated.", "success");
        updateChrome();
      } catch (error) {
        state.trumedia = { ...state.trumedia, connected: false, status: error.code === "expired_session" ? "expired" : state.trumedia.status };
        status.textContent = error.message;
        updateChrome();
      } finally {
        revalidate.disabled = false;
        revalidate.textContent = "Revalidate current";
      }
    } },
  });
  clear(workflowDialogContent).append(
    workflowHeader("Secure integration", "TruMedia session", "Configure headless access before selecting or processing a game."),
    element("div", { className: "workflow-body" }, [
      element("div", { className: `integration-summary ${state.trumedia.connected ? "connected" : "needs-auth"}` }, [
        element("span", { className: "integration-dot", "aria-hidden": "true" }),
        element("div", {}, [
          element("strong", { text: state.trumedia.connected ? "Session configured" : `Session ${statusLabel(state.trumedia.status || "missing")}` }),
          element("span", { text: state.trumedia.connected ? "Automatic game and pitch discovery is available." : "Upload a current session before running a live game." }),
        ]),
      ]),
      element("ol", { className: "setup-steps" }, [
        element("li", { text: "Run python src/trumedia_auth.py from the repository root." }),
        element("li", { text: "Sign in and wait for the TruMedia baseball workspace to load." }),
        element("li", { text: "Return to the terminal and press Enter to export the session." }),
        element("li", { text: "Enter CATCHER_STANCE_ADMIN_TOKEN below." }),
        element("li", { text: "Upload data/auth/playwright_state.export.json." }),
      ]),
      element("div", { className: "security-note" }, [
        element("strong", { text: "Bearer credential" }),
        element("span", { text: "The uploaded file is encrypted in transit when deployed behind HTTPS and stored with owner-only permissions." }),
      ]),
      element("label", { className: "field" }, [element("span", { text: "Admin token" }), token]),
      element("label", { className: "field" }, [element("span", { text: "Playwright session JSON" }), file]),
      status,
      element("div", { className: "dialog-actions" }, [
        element("button", { className: "button ghost", type: "button", text: "Cancel", on: { click: closeWorkflow } }),
        ...(state.trumedia.connected ? [revalidate] : []),
        submit,
      ]),
    ]),
  );
  workflowDialog.showModal();
}

function openMatchSelection(game, candidates, container, reprocess = false) {
  let selected = candidates[0]?.id || "";
  const options = candidates.map((candidate, index) => element("label", { className: "candidate-card" }, [
    element("input", {
      type: "radio", name: "candidate", value: candidate.id, checked: index === 0,
      on: { change: () => { selected = candidate.id; } },
    }),
    element("span", {}, [
      element("strong", { text: candidate.opponent || "TruMedia game" }),
      element("small", {
        text: [
          formatDate(candidate.date),
          candidate.start_time || null,
          candidate.game_number ? `Game ${candidate.game_number}` : null,
          candidate.result || null,
        ].filter(Boolean).join(" · "),
      }),
    ]),
  ]));
  const confirm = element("button", {
    className: "button primary", type: "button", text: "Use selected game", disabled: !selected,
    on: { click: async () => {
      confirm.disabled = true;
      confirm.textContent = "Confirming…";
      try {
        await confirmTruMediaMatch(game.id, selected);
        closeWorkflow();
        await submitRun(game, container, reprocess);
      } catch (error) {
        toast(error.message, "danger");
        confirm.disabled = false;
        confirm.textContent = "Try again";
      }
    } },
  });
  clear(workflowDialogContent).append(
    workflowHeader("Match required", "Choose the TruMedia game", "The schedule match was not unique. Confirm the exact game before downloading pitches."),
    element("div", { className: "workflow-body" }, [
      element("div", { className: "candidate-list" }, options.length ? options : [element("p", { className: "empty-state", text: "No candidates were found for this date." })]),
      element("div", { className: "dialog-actions" }, [
        element("button", { className: "button ghost", type: "button", text: "Cancel", on: { click: closeWorkflow } }),
        confirm,
      ]),
    ]),
  );
  workflowDialog.showModal();
}

function updateChrome() {
  const count = activeRuns().length;
  activityText.textContent = count ? `${count} active ${count === 1 ? "run" : "runs"}` : "No active runs";
  activityButton.classList.toggle("is-active", count > 0);
  connectionState.textContent = state.connected
    ? `Live · ${state.lastUpdated?.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) || "now"}`
    : "Reconnecting";
  connectionState.classList.toggle("is-offline", !state.connected);
  trumediaText.textContent = state.trumedia.connected ? "TruMedia ready" : "Connect TruMedia";
  trumediaButton.classList.toggle("connected", Boolean(state.trumedia.connected));
  trumediaButton.setAttribute(
    "aria-label",
    state.trumedia.connected ? "Manage connected TruMedia session" : "Connect TruMedia session",
  );

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
  const stages = progress.stages || {};
  const elapsed = ACTIVE.has(run.status) && Number(run.created_at)
    ? Date.now() / 1000 - Number(run.created_at)
    : Number(run.performance?.elapsed_seconds);
  const rate = run.performance?.pitches_per_minute == null
    ? null
    : Number(run.performance.pitches_per_minute);
  const eta = run.performance?.eta_seconds == null
    ? null
    : Number(run.performance.eta_seconds);
  const activeStage = progress.active_stage || progress.phase;
  const facts = [
    Number.isFinite(elapsed) ? `${Math.floor(elapsed / 60)}m ${Math.round(elapsed % 60)}s elapsed` : null,
    Number.isFinite(rate) ? `${rate.toFixed(1)} pitches/min` : null,
    Number.isFinite(eta) ? `about ${Math.max(1, Math.ceil(eta / 60))}m remaining` : null,
  ].filter(Boolean).join(" · ");
  return element("div", { className: "run-progress", dataset: { runProgress: run.id } }, [
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
    element("div", { className: "stage-strip", "aria-label": "Pipeline stages" },
      PIPELINE_STAGES.map((stage) => {
        const item = stages[stage] || {};
        const stagePercent = Number(item.percent);
        const stateName = item.status === "complete" || stagePercent >= 100
          ? "complete"
          : activeStage === stage
            ? "active"
            : item.current !== undefined
              ? "started"
              : "pending";
        return element("span", { className: `stage-chip ${stateName}` }, [
          element("i", { "aria-hidden": "true" }),
          element("span", { text: statusLabel(stage.replace("_pitches", "")) }),
          ...(Number.isFinite(stagePercent) ? [element("strong", { text: `${stagePercent.toFixed(0)}%` })] : []),
        ]);
      }),
    ),
    ...(facts ? [element("small", { className: "progress-facts", text: facts })] : []),
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
  const node = badge(run.read_only ? "Sample" : statusLabel(run.status), run.read_only ? "sample" : variant);
  node.dataset.runBadge = run.id;
  return node;
}

function gameCard(game) {
  const run = latestRunForGame(game.id);
  const selected = state.selectedGameId === game.id;
  return element("button", {
    className: `schedule-card${selected ? " selected" : ""}`,
    dataset: { gameId: game.id },
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
    const matchesSite = state.siteFilter === "all" || game.site === state.siteFilter;
    const matchesResult = state.resultFilter === "all" || game.result?.startsWith(state.resultFilter);
    const matchesConference = state.conferenceFilter === "all" || String(game.conference) === state.conferenceFilter;
    const matchesFrom = !state.dateFrom || game.date >= state.dateFrom;
    const matchesTo = !state.dateTo || game.date <= state.dateTo;
    return matchesText && matchesStatus && matchesSite && matchesResult && matchesConference && matchesFrom && matchesTo;
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
    ["all", "All games"], ["complete", "Analyzed"], ["active", "Active"], ["not-run", "Not analyzed"],
  ].map(([value, text]) => element("option", { value, text, selected: state.gameFilter === value })));

  const section = element("section", { className: "page" });
  const seasonSelect = element("select", {
    className: "season-select", "aria-label": "Schedule season",
    on: { change: async (event) => {
      const previousSeason = state.season;
      state.season = Number(event.target.value);
      state.selectedGameId = null;
      try {
        state.schedule = await getSchedule(state.season);
        renderGames();
      } catch (error) {
        state.season = previousSeason;
        toast(error.message, "danger");
        renderGames();
      }
    } },
  }, state.seasons.map((year) => element("option", { value: year, text: `${year} season`, selected: year === state.season })));
  section.append(pageHeader(
    `${state.season} Duke Baseball`,
    "Game workspace",
    "Search completed games, connect them to TruMedia, and move directly into pitch-level review.",
    [seasonSelect],
  ));
  const layout = element("div", { className: "games-layout" });
  const rail = element("aside", { className: "schedule-panel", "aria-label": "Game schedule" }, [
    element("div", { className: "panel-toolbar" }, [search, filter]),
    element("details", { className: "filter-drawer" }, [
      element("summary", { text: "More filters" }),
      element("div", { className: "filter-grid" }, [
        element("select", { "aria-label": "Filter by site", on: { change: (event) => { state.siteFilter = event.target.value; renderGames(); } } },
          [["all", "All sites"], ["home", "Home"], ["away", "Away"], ["neutral", "Neutral"]].map(([value, text]) => element("option", { value, text, selected: state.siteFilter === value }))),
        element("select", { "aria-label": "Filter by result", on: { change: (event) => { state.resultFilter = event.target.value; renderGames(); } } },
          [["all", "Wins and losses"], ["W", "Wins"], ["L", "Losses"]].map(([value, text]) => element("option", { value, text, selected: state.resultFilter === value }))),
        element("select", { "aria-label": "Filter by conference", on: { change: (event) => { state.conferenceFilter = event.target.value; renderGames(); } } },
          [["all", "All opponents"], ["true", "Conference"], ["false", "Non-conference"]].map(([value, text]) => element("option", { value, text, selected: state.conferenceFilter === value }))),
        element("input", { type: "date", value: state.dateFrom, "aria-label": "Games from date", on: { change: (event) => { state.dateFrom = event.target.value; renderGames(); } } }),
        element("input", { type: "date", value: state.dateTo, "aria-label": "Games through date", on: { change: (event) => { state.dateTo = event.target.value; renderGames(); } } }),
      ]),
    ]),
    element("div", { className: "schedule-list" },
      filtered.length ? filtered.map(gameCard) : [element("div", { className: "empty-state", text: "No games match these filters." })]),
  ]);

  const detail = element("article", { className: "game-detail" });
  if (!game) {
    detail.append(element("div", { className: "empty-state", text: "Select a game to continue." }));
  } else {
    const runAction = run?.status === "complete"
      ? element("div", { className: "hero-actions" }, [
          element("a", { className: "button primary", href: `#/results/${run.id}`, text: "Review results" }),
          ...(!run.read_only ? [element("button", {
            className: "button hero-secondary", type: "button", text: "Reprocess",
            on: { click: () => submitRun(game, detail, true) },
          })] : []),
        ])
      : run && ACTIVE.has(run.status) && run.result_count > 0
        ? element("div", { className: "hero-actions" }, [
            element("a", { className: "button primary", href: `#/results/${run.id}`, text: `Review ${run.result_count} available` }),
            element("span", { className: "muted", text: "Processing continues in the background" }),
          ])
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
        element("span", { text: run ? `${run.result_count || 0} pitch results` : "No analysis yet", dataset: run ? { resultCount: run.id } : undefined }),
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
      element("summary", { text: "Storage settings" }),
      element("div", { className: "settings-grid" }, [
        element("label", { className: "check-row" }, [
          element("input", { id: "retainSources", type: "checkbox" }),
          element("span", { text: "Retain full source clips after compact reviews are created" }),
        ]),
        element("p", { className: "muted", text: "By default, full downloads are removed only after results and review clips validate." }),
      ]),
    ]);
    if (!run?.read_only) detail.append(advanced);
    if (!run?.read_only) {
      detail.append(element("button", {
        className: `integration-strip ${state.trumedia.connected ? "connected" : "needs-auth"}`,
        type: "button",
        on: { click: () => openAuthentication() },
      }, [
        element("span", { className: "integration-dot", "aria-hidden": "true" }),
        element("div", {}, [
          element("strong", { text: state.trumedia.connected ? "TruMedia session configured" : "TruMedia session required" }),
          element("small", { text: state.trumedia.connected ? "Automatic matching is ready; ambiguous games require confirmation." : "Upload a protected Playwright session before downloading." }),
        ]),
      ]));
    }
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

async function submitRun(game, container, reprocess = false) {
  const button = container.querySelector(reprocess ? ".hero-secondary" : ".button.primary");
  const retainSources = container.querySelector("#retainSources")?.checked || false;
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    const run = reprocess
      ? await reprocessRun(game.id, { retain_sources: retainSources })
      : await startRun({ game_id: game.id, retain_sources: retainSources });
    state.runs = [run, ...state.runs.filter((item) => item.id !== run.id)];
    liveRegion.textContent = `Run started for ${game.opponent}`;
    toast("Run started. You can continue using the application.", "success");
    location.hash = "#/runs";
    schedulePoll(500);
  } catch (error) {
    if (error.code === "auth_required") {
      button.disabled = false;
      button.textContent = "Connect TruMedia";
      openAuthentication(game, container, reprocess);
      return;
    }
    if (error.code === "match_required") {
      button.disabled = false;
      button.textContent = "Choose TruMedia game";
      openMatchSelection(game, error.candidates || [], container, reprocess);
      return;
    }
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
    [element("button", { className: "button ghost", type: "button", text: "Refresh", on: { click: () => refreshRuns(true, true) } })],
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
    const card = element("article", { className: `run-card${ACTIVE.has(run.status) ? " active-run" : ""}`, dataset: { runId: run.id } }, [
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
        element("span", { text: `${run.result_count || 0} results`, dataset: { resultCount: run.id } }),
        element("span", { text: `${(run.manifest?.downloaded || 0) + (run.manifest?.cleaned || 0)}/${run.manifest?.total || 0} clips downloaded` }),
        ...(run.revision ? [element("span", { text: `Revision ${run.revision}` })] : []),
        ...(run.cleanup?.status ? [element("span", { text: `Storage: ${statusLabel(run.cleanup.status)}` })] : []),
        element("span", { text: run.read_only ? "Read-only fixture" : "Live run" }),
      ]),
      run.status === "complete" || run.result_count > 0
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

function metricsGrid(run) {
  return element("div", { className: "metrics-grid" },
    resultSummary(run.results || []).map(([label, value, note]) =>
      element("article", { className: "metric-card" }, [
        element("span", { text: label }),
        element("strong", { text: value }),
        element("small", { text: note }),
      ]),
    ),
  );
}

function filteredResultRows(run) {
  const query = state.resultSearch.toLowerCase();
  return (run.results || []).filter((row) => {
    const textMatch = `${row.clip_id} ${row.stance} ${row.status}`.toLowerCase().includes(query);
    const stanceMatch = state.stanceFilter === "all" ||
      (state.stanceFilter === "abstained" ? !row.accepted : row.stance === state.stanceFilter);
    const qualityMatch = state.qualityFilter === "all" ||
      (state.qualityFilter === "flagged" && (row.quality_flags?.length || !row.accepted)) ||
      (state.qualityFilter === "clear" && row.accepted && !row.quality_flags?.length);
    return textMatch && stanceMatch && qualityMatch;
  });
}

function resultRow(row, run) {
  const stance = row.stance || "Abstained";
  const quality = row.quality_flags?.length ? "Review" : row.accepted ? "Clear" : "Rejected";
  return element("button", {
    className: "result-row",
    dataset: { clipId: row.clip_id },
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

function openGameTracker(run) {
  const sheetUrl = element("input", { type: "url", placeholder: "https://docs.google.com/spreadsheets/d/…", "aria-label": "Google Sheets link" });
  const tab = element("select", { disabled: true, "aria-label": "GameTracker sheet tab" }, [element("option", { text: "Connect a sheet first" })]);
  const connectStatus = element("p", { className: "form-status", text: "Prototype only: no request will be sent to Google." });
  const exportButton = element("button", { className: "button sheets-button", type: "button", text: "Simulate export", disabled: true });
  const connectButton = element("button", {
    className: "button secondary", type: "button", text: "Preview sheet tabs",
    on: { click: () => {
      if (!/^https:\/\/docs\.google\.com\/spreadsheets\/d\/[A-Za-z0-9_-]+/.test(sheetUrl.value)) {
        connectStatus.textContent = "Enter a valid Google Sheets URL to continue.";
        return;
      }
      connectButton.disabled = true;
      connectButton.textContent = "Loading simulated tabs…";
      connectStatus.textContent = "Demonstrating the future tab-discovery step. No network request is being made.";
      window.setTimeout(() => {
        tab.replaceChildren(...["GameTracker", "Pitch Log", "Catcher Review"].map((name) => element("option", { value: name, text: name })));
        tab.disabled = false;
        exportButton.disabled = false;
        connectButton.textContent = "Tabs previewed";
      }, 650);
    } },
  });
  exportButton.addEventListener("click", () => {
    exportButton.disabled = true;
    exportButton.textContent = "Simulating…";
    window.setTimeout(() => {
      connectStatus.textContent = `Simulation complete: ${run.result_count || 0} pitches would be sent to “${tab.value}”. No data was written.`;
      exportButton.textContent = "Simulation complete";
    }, 700);
  });
  clear(workflowDialogContent).append(
    workflowHeader("GameTracker prototype", "Export to GameTracker", "Preview the planned Google Sheets workflow before the integration is enabled."),
    element("div", { className: "prototype-banner" }, [
      element("strong", { text: "Simulation" }),
      element("span", { text: "This dialog does not read, store, or modify a Google Sheet." }),
    ]),
    element("div", { className: "workflow-body" }, [
      element("label", { className: "field" }, [element("span", { text: "Google Sheet link" }), sheetUrl]),
      connectButton,
      element("label", { className: "field" }, [element("span", { text: "Exact destination tab" }), tab]),
      element("fieldset", { className: "option-fieldset" }, [
        element("legend", { text: "Write mode" }),
        element("label", { className: "check-row" }, [element("input", { type: "radio", name: "write-mode", checked: true }), element("span", { text: "Append pitch rows" })]),
        element("label", { className: "check-row" }, [element("input", { type: "radio", name: "write-mode" }), element("span", { text: "Replace this game's rows" })]),
      ]),
      element("label", { className: "check-row auto-export" }, [element("input", { type: "checkbox" }), element("span", { text: "Automatically export after future completed runs" })]),
      connectStatus,
      element("div", { className: "dialog-actions" }, [
        element("button", { className: "button ghost", type: "button", text: "Close", on: { click: closeWorkflow } }),
        exportButton,
      ]),
    ]),
  );
  workflowDialog.showModal();
}

function renderResults(runId) {
  const run = state.runDetails.get(runId) || state.runs.find((item) => item.id === runId);
  if (!run) {
    clear(content).append(element("section", { className: "page" }, [
      pageHeader("Results", "Run not found", "The requested run is unavailable."),
      element("a", { className: "button primary", href: "#/runs", text: "Back to runs" }),
    ]));
    return;
  }
  const rows = filteredResultRows(run);

  const section = element("section", { className: "page results-page" });
  section.append(pageHeader(
    run.read_only ? "Sample analysis" : run.results_complete ? "Completed analysis" : "Live analysis",
    `${run.game?.opponent || "Run"} results`,
    `${formatDate(run.game?.date)} · ${run.result_count} pitch-level classifications`,
    [
      element("button", { className: "button sheets-button", type: "button", text: "Export to GameTracker", on: { click: () => openGameTracker(run) } }),
      run.results_complete
        ? element("a", { className: "button ghost", href: apiUrl(`/api/results/${run.id}/json`), text: "JSON", download: "" })
        : element("button", { className: "button ghost", type: "button", text: "JSON after completion", disabled: true }),
      run.results_complete
        ? element("a", { className: "button secondary", href: apiUrl(`/api/results/${run.id}/csv`), text: "Export CSV", download: "" })
        : element("button", { className: "button secondary", type: "button", text: "CSV after completion", disabled: true }),
    ],
  ));
  if (!run.results_complete && ACTIVE.has(run.status)) section.append(progressBar(run));
  section.append(metricsGrid(run));

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
      element("span", { className: "muted results-visible-count", text: `${rows.length} of ${run.results?.length || 0} pitches` }),
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
    if (row.media_mode !== "review" && Number.isFinite(Number(row.window_start_seconds))) video.currentTime = Number(row.window_start_seconds);
  }, { once: true });
  const media = element("div", { className: "pitch-media" }, [video]);
  const sourceButton = element("button", {
    className: "segmented active",
    type: "button",
    text: row.media_mode === "review" ? "Compact review" : "Source clip",
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

function patchRunNodes(run) {
  document.querySelectorAll(`[data-run-progress="${run.id}"]`).forEach((node) => {
    node.replaceWith(progressBar(run));
  });
  document.querySelectorAll(`[data-run-badge="${run.id}"]`).forEach((node) => {
    node.replaceWith(runBadge(run));
  });
  document.querySelectorAll(`[data-result-count="${run.id}"]`).forEach((node) => {
    node.textContent = node.closest(".score-strip")
      ? `${run.result_count || 0} pitch results`
      : `${run.result_count || 0} results`;
  });
  const card = document.querySelector(`[data-run-id="${run.id}"]`);
  if (card) card.classList.toggle("active-run", ACTIVE.has(run.status));
}

function patchResults(run) {
  state.runDetails.set(run.id, run);
  const current = route();
  if (current.view !== "results" || current.id !== run.id) return;
  const oldMetrics = content.querySelector(".metrics-grid");
  if (oldMetrics) oldMetrics.replaceWith(metricsGrid(run));
  const list = content.querySelector(".result-list");
  if (list) {
    const rows = filteredResultRows(run);
    const existing = new Set(
      [...list.querySelectorAll("[data-clip-id]")].map((node) => node.dataset.clipId),
    );
    if (rows.length && !existing.size) list.replaceChildren();
    for (const row of rows) {
      if (!existing.has(row.clip_id)) list.append(resultRow(row, run));
    }
    if (!rows.length) list.replaceChildren(element("div", { className: "empty-state", text: "No pitches match these filters." }));
    const count = content.querySelector(".results-visible-count");
    if (count) count.textContent = `${rows.length} of ${run.results?.length || 0} pitches`;
  }
  patchRunNodes(run);
}

async function ensureRunDetail(runId, signal) {
  const detail = await getRun(runId, signal ? { signal } : undefined);
  state.runDetails.set(runId, detail);
  return detail;
}

function mergeRunSummaries(incoming) {
  const previous = new Map(state.runs.map((run) => [run.id, run]));
  const next = incoming || [];
  const structural = next.length !== state.runs.length || next.some((run) => !previous.has(run.id));
  const terminalTransition = next.some((run) => {
    const old = previous.get(run.id);
    return old && ACTIVE.has(old.status) !== ACTIVE.has(run.status);
  });
  state.runs = next;
  return { previous, structural, terminalTransition };
}

async function applyRunSummaries(incoming) {
  const { previous, structural, terminalTransition } = mergeRunSummaries(incoming);
  updateChrome();
  const current = route();
  if (structural || terminalTransition) {
    if (current.view === "results" && current.id) {
      try {
        await ensureRunDetail(current.id);
      } catch (_) {
        // The normal render below will show the missing-run state.
      }
    }
    render();
    return;
  }
  for (const run of state.runs) patchRunNodes(run);
  if (current.view === "results" && current.id) {
    const summary = state.runs.find((run) => run.id === current.id);
    const old = previous.get(current.id);
    if (summary && (!old || summary.result_count !== old.result_count || summary.updated_at !== old.updated_at)) {
      patchResults(await ensureRunDetail(current.id));
    }
  }
}

async function refreshRuns(announce = false, replaceInFlight = false) {
  if (state.pollController) {
    if (!replaceInFlight) return;
    state.pollController.abort();
  }
  const controller = new AbortController();
  state.pollController = controller;
  try {
    const response = await getRunSummaries(state.runsEtag, controller.signal);
    if (!response.notModified) {
      state.runsEtag = response.etag;
      await applyRunSummaries(response.payload.runs || []);
    } else {
      for (const run of state.runs) patchRunNodes(run);
    }
    state.connected = true;
    state.lastUpdated = new Date();
    state.pollFailures = 0;
    updateChrome();
    if (announce) toast("Run status refreshed.", "success");
  } catch (error) {
    if (error.name === "AbortError") return;
    state.connected = false;
    state.pollFailures += 1;
    updateChrome();
    if (announce) toast(error.message, "danger");
  } finally {
    if (state.pollController === controller) {
      state.pollController = null;
      schedulePoll();
    }
  }
}

function schedulePoll(delay) {
  window.clearTimeout(state.timer);
  if (document.visibilityState === "hidden") return;
  const activeDelay = activeRuns().length ? 3000 : 30000;
  const backoff = Math.min(30000, activeDelay * 2 ** state.pollFailures);
  state.timer = window.setTimeout(() => refreshRuns(false), delay ?? backoff);
}

async function routeChanged() {
  const current = route();
  if (current.view === "results" && current.id) {
    try {
      await ensureRunDetail(current.id);
    } catch (_) {
      state.runDetails.delete(current.id);
    }
  }
  render();
}

async function bootstrap() {
  try {
    const [schedule, seasonsPayload, trumedia, summaries] = await Promise.all([
      getSchedule(state.season),
      getSeasons(),
      getTruMediaStatus(),
      getRunSummaries(),
    ]);
    state.schedule = schedule;
    state.seasons = seasonsPayload.seasons || [2026];
    state.trumedia = trumedia;
    state.runs = summaries.payload?.runs || [];
    state.runsEtag = summaries.etag;
    state.connected = true;
    state.lastUpdated = new Date();
    await routeChanged();
  } catch (error) {
    state.connected = false;
    toast(error.message, "danger");
    render();
  } finally {
    schedulePoll();
  }
}

window.addEventListener("hashchange", routeChanged);
window.addEventListener("focus", () => refreshRuns(false, true));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") refreshRuns(false, true);
  else {
    window.clearTimeout(state.timer);
    state.pollController?.abort();
  }
});
activityButton.addEventListener("click", () => { location.hash = "#/runs"; });
trumediaButton.addEventListener("click", () => openAuthentication());
pitchDialog.addEventListener("close", () => clear(pitchDialogContent));
pitchDialog.addEventListener("click", (event) => {
  if (event.target === pitchDialog) pitchDialog.close();
});
workflowDialog.addEventListener("close", () => clear(workflowDialogContent));
workflowDialog.addEventListener("click", (event) => {
  if (event.target === workflowDialog) workflowDialog.close();
});

bootstrap();
