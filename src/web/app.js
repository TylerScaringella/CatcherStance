import {
  apiUrl,
  confirmTruMediaMatch,
  controlRun,
  deleteRun,
  getRun,
  getRunSummaries,
  getSchedule,
  getScheduleSettings,
  getSeasons,
  getTruMediaStatus,
  purgeSourceClips,
  refreshSchedule,
  recoverPitchSource,
  savePitchReview,
  reprocessRun,
  setActiveSeason,
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
  season: null,
  activeSeason: null,
  seasons: [],
  scheduleSettings: null,
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

function stanceKey(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized.includes("left") || normalized === "lkd") return "LKD";
  if (normalized.includes("right") || normalized === "rkd") return "RKD";
  if (normalized.includes("squat")) return "Squat";
  return "Abstained";
}

function stanceLabel(value) {
  return {
    LKD: "Knee-down Left",
    RKD: "Knee-down Right",
    Squat: "Squat",
    Abstained: "Abstained",
  }[stanceKey(value)] || String(value || "Abstained");
}

function reviewStance(value) {
  const key = stanceKey(value);
  return { LKD: "KneeDown - Left", RKD: "KneeDown - Right", Squat: "Squat" }[key] || "";
}

function resultAttention(row) {
  return Boolean(row.attention_required);
}

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

function openAdminUnlock(onUnlocked) {
  const token = element("input", { type: "password", autocomplete: "current-password", placeholder: "Admin token", "aria-label": "Admin token" });
  const status = element("p", { className: "form-status", text: "Review changes require the configured admin token." });
  const unlock = element("button", {
    className: "button primary", type: "button", text: "Unlock review actions",
    on: { click: async () => {
      if (!token.value) {
        status.textContent = "Enter the admin token to continue.";
        return;
      }
      unlock.disabled = true;
      status.textContent = "Unlocking protected review actions…";
      try {
        await unlockAdmin(token.value);
        closeWorkflow();
        await onUnlocked();
      } catch (error) {
        status.textContent = error.message;
        unlock.disabled = false;
      }
    } },
  });
  clear(workflowDialogContent).append(
    workflowHeader("Protected action", "Unlock pitch review", "Enter the admin token configured for this server. No TruMedia upload is required to save a review."),
    element("div", { className: "workflow-body" }, [
      element("label", { className: "field" }, [element("span", { text: "Admin token" }), token]),
      status,
      element("div", { className: "dialog-actions" }, [
        element("button", { className: "button ghost", type: "button", text: "Cancel", on: { click: closeWorkflow } }),
        unlock,
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

function workspaceContext(section, detail = "2026 season") {
  return element("div", { className: "workspace-context", "aria-label": "Workspace context" }, [
    element("span", { className: "workspace-context-brand", text: "DUKE BASEBALL / OPERATIONS" }),
    element("span", { className: "workspace-context-separator", "aria-hidden": "true", text: "/" }),
    element("span", { text: section }),
    element("span", { className: "workspace-context-detail", text: detail }),
  ]);
}

function scheduleFreshness(settings = state.scheduleSettings, schedule = state.schedule) {
  if (schedule?.unpublished) return `Duke has not published the ${schedule.season} schedule yet`;
  if (settings?.error) return `Needs refresh · ${settings.error}`;
  if (schedule?.stale || settings?.stale) return "Using cached schedule · refresh recommended";
  if (settings?.last_successful_refresh_at) return `Updated ${formatDateTime(settings.last_successful_refresh_at)}`;
  return "Not refreshed yet";
}

async function loadScheduleForSeason(season) {
  try {
    return await getSchedule(season);
  } catch (error) {
    if (error.code === "schedule_unpublished" || error.code === "schedule_unavailable") {
      return {
        team: "Duke",
        team_id: "duke",
        sport: "Baseball",
        season: Number(season),
        games: [],
        unpublished: true,
        error: error.message,
      };
    }
    throw error;
  }
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function progressBar(run) {
  const progress = run.progress || {};
  const percent = progress.percent == null ? null : Number(progress.percent);
  const determinate = percent != null && Number.isFinite(percent);
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
        const stagePercent = item.percent == null ? null : Number(item.percent);
        const stateName = item.status === "complete" || stagePercent >= 100
          ? "complete"
          : activeStage === stage
            ? "active"
            : item.current != null
              ? "started"
              : "pending";
        return element("span", { className: `stage-chip ${stateName}` }, [
          element("i", { "aria-hidden": "true" }),
          element("span", { text: statusLabel(stage.replace("_pitches", "")) }),
          ...(stagePercent != null && Number.isFinite(stagePercent) ? [element("strong", { text: `${stagePercent.toFixed(0)}%` })] : []),
        ]);
      }),
    ),
    ...(facts ? [element("small", { className: "progress-facts", text: facts })] : []),
  ]);
}

function gameRunAction(game, run) {
  if (game.status !== "completed") {
    return element("div", { className: "hero-actions" }, [
      badge("Scheduled", "neutral"),
      element("span", { className: "muted", text: "Analysis becomes available after the game is completed." }),
    ]);
  }
  if (run?.status === "complete") {
    return element("div", { className: "hero-actions", dataset: { runAction: run.id, actionKind: "game", actionState: "complete" } }, [
      element("a", { className: "button primary", href: `#/results/${run.id}`, text: "Review results" }),
      ...(!run.read_only ? [element("button", {
        className: "button hero-secondary", type: "button", text: "Reprocess",
        on: { click: (event) => submitRun(game, event.currentTarget.closest(".game-detail"), true) },
      })] : []),
    ]);
  }
  if (run && ACTIVE.has(run.status) && run.result_count > 0) {
    return element("div", { className: "hero-actions", dataset: { runAction: run.id, actionKind: "game", actionState: `partial-${run.result_count}` } }, [
      element("a", { className: "button primary", href: `#/results/${run.id}`, text: `Review ${run.result_count} available` }),
      element("span", { className: "muted", text: "Processing continues in the background" }),
    ]);
  }
  return element("button", {
    className: "button primary",
    dataset: run ? {
      runAction: run.id,
      actionKind: "game",
      actionState: ACTIVE.has(run.status) ? "active" : run.status || "ready",
    } : undefined,
    type: "button",
    disabled: run && ACTIVE.has(run.status),
    text: run && ACTIVE.has(run.status) ? "Run in progress" : run?.status === "ready" ? "Run detection" : "Start detection",
    on: { click: (event) => submitRun(game, event.currentTarget.closest(".game-detail")) },
  });
}

function runCardAction(run) {
  const open = run.status === "complete" || run.result_count > 0
    ? element("a", { className: "button secondary", href: `#/results/${run.id}`, text: "Open results" })
    : element("a", { className: "button ghost", href: "#/games", text: "View game" });
  const controls = [];
  if (ACTIVE.has(run.status)) {
    controls.push(
      element("button", { className: "button ghost", type: "button", text: "Pause", on: { click: () => requestRunControl(run, "pause") } }),
      element("button", { className: "button ghost", type: "button", text: "Stop", on: { click: () => requestRunControl(run, "stop") } }),
      element("button", { className: "button danger-outline", type: "button", text: "Cancel", on: { click: () => requestRunControl(run, "cancel") } }),
    );
  } else if (run.status === "paused" || run.status === "stopped") {
    controls.push(element("button", { className: "button secondary", type: "button", text: "Resume", on: { click: () => resumeRun(run) } }));
  }
  if (run.status === "complete" || run.status === "failed" || run.status === "interrupted" || run.status === "stopped" || run.status === "canceled") {
    controls.push(element("button", { className: "button danger-outline", type: "button", text: "Delete", on: { click: () => deleteRunWithConfirmation(run) } }));
  }
  return element("div", {
    className: "run-actions",
    tabindex: "0",
    role: "group",
    "aria-label": `Actions for run ${run.id}`,
    on: {
      click: (event) => {
        if (event.target.closest("button, a")) return;
        window.location.hash = run.result_count > 0 ? `#/results/${run.id}` : "#/games";
      },
    },
    dataset: { runAction: run.id, actionKind: "card", actionState: `${run.status}-${run.result_count || 0}` },
  }, [open, ...controls]);
}

async function requestRunControl(run, action) {
  const labels = { pause: "pause", stop: "stop after the current pitch", cancel: "cancel" };
  if (!window.confirm(`Confirm ${labels[action]} this run?`)) return;
  try {
    await controlRun(run.id, action);
    toast(`${action[0].toUpperCase()}${action.slice(1)} requested.`, "info");
    await refreshRuns(true, true);
  } catch (error) {
    toast(error.message, "danger");
  }
}

async function resumeRun(run) {
  try {
    const resumed = await startRun({ game_id: run.game?.id });
    state.runs = [resumed, ...state.runs.filter((item) => item.id !== resumed.id)];
    toast("Run resumed.", "success");
    schedulePoll(500);
  } catch (error) {
    toast(error.message, "danger");
  }
}

async function deleteRunWithConfirmation(run) {
  if (!window.confirm(`Delete the finished run for ${run.game?.opponent || run.id}? This removes its videos, evidence, and exports permanently.`)) return;
  openAdminUnlock(async () => {
    try {
      await deleteRun(run.id);
      state.runs = state.runs.filter((item) => item.id !== run.id);
      state.runDetails.delete(run.id);
      toast("Run deleted.", "success");
      if (route().view === "results" && route().id === run.id) location.hash = "#/runs";
      else renderRuns();
    } catch (error) {
      toast(error.message, "danger");
    }
  });
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
    element("span", { className: "schedule-date", text: formatDate(game.date).replace(/, \d{4}/, "") }),
    element("span", { className: "schedule-opponent", text: game.opponent }),
    element("span", { className: "schedule-meta", text: `${game.location} · ${game.result || "Scheduled"}` }),
    element("span", { className: "schedule-link", text: run ? `${run.result_count || 0} pitches` : game.status === "completed" ? "Ready for analysis" : "Upcoming" }),
    run ? runBadge(run) : badge("Not run"),
  ]);
}

function openScheduleManagement() {
  const token = element("input", { type: "password", autocomplete: "current-password", placeholder: "Admin token", "aria-label": "Admin token" });
  const season = element("select", { "aria-label": "Schedule management season" },
    state.seasons.map((year) => element("option", { value: year, text: `${year} season`, selected: year === state.season })));
  const status = element("p", { className: "form-status", text: scheduleFreshness() });
  const setButton = element("button", {
    className: "button secondary", type: "button", text: "Set as current",
    on: { click: async () => {
      setButton.disabled = true;
      status.textContent = "Saving active season…";
      try {
        await unlockAdmin(token.value);
        state.scheduleSettings = await setActiveSeason(season.value);
        state.activeSeason = state.scheduleSettings.active_season;
        state.season = Number(season.value);
        state.selectedGameId = null;
        state.schedule = await loadScheduleForSeason(state.season);
        toast(`${state.activeSeason} is now the active season.`, "success");
        status.textContent = scheduleFreshness();
        renderGames();
      } catch (error) {
        status.textContent = error.message;
      } finally {
        setButton.disabled = false;
      }
    } },
  });
  const refreshButton = element("button", {
    className: "button primary", type: "button", text: "Refresh from Duke",
    on: { click: async () => {
      refreshButton.disabled = true;
      status.textContent = "Fetching the official Duke schedule…";
      try {
        await unlockAdmin(token.value);
        const response = await refreshSchedule(season.value);
        state.scheduleSettings = response.settings;
        if (Number(season.value) === state.season) state.schedule = response.schedule;
        toast(
          response.schedule.stale ? `${season.value} schedule is unchanged; cached data is being shown.` : `${season.value} schedule refreshed.`,
          response.schedule.stale ? "info" : "success",
        );
        status.textContent = scheduleFreshness(response.settings, response.schedule);
        renderGames();
      } catch (error) {
        status.textContent = error.code === "schedule_unpublished"
          ? `Duke has not published the ${season.value} schedule yet.`
          : error.message;
      } finally {
        refreshButton.disabled = false;
      }
    } },
  });
  clear(workflowDialogContent).append(
    workflowHeader("Schedule administration", "Duke Baseball schedule", "Choose the active season or fetch the latest official schedule. Schedule changes require the existing admin token."),
    element("div", { className: "workflow-body" }, [
      element("label", { className: "field" }, [element("span", { text: "Season" }), season]),
      element("label", { className: "field" }, [element("span", { text: "Admin token" }), token]),
      element("div", { className: "schedule-freshness", "aria-live": "polite" }, [status]),
      element("div", { className: "dialog-actions" }, [
        element("button", { className: "button ghost", type: "button", text: "Cancel", on: { click: closeWorkflow } }),
        setButton,
        refreshButton,
      ]),
    ]),
  );
  workflowDialog.showModal();
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
  section.append(workspaceContext("Game operations", `${state.season} season`));
  const seasonSelect = element("select", {
    className: "season-select", "aria-label": "Schedule season",
    on: { change: async (event) => {
      const previousSeason = state.season;
      state.season = Number(event.target.value);
      state.selectedGameId = null;
      try {
        state.schedule = await loadScheduleForSeason(state.season);
        renderGames();
      } catch (error) {
        state.season = previousSeason;
        toast(error.message, "danger");
        renderGames();
      }
    } },
  }, state.seasons.map((year) => element("option", { value: year, text: `${year} season`, selected: year === state.season })));
  const scheduleButton = element("button", {
    className: "button ghost", type: "button", text: "Manage schedule",
    on: { click: openScheduleManagement },
  });
  section.append(pageHeader(
    `${state.season} Duke Baseball`,
    "Game workspace",
    "Search the official schedule, connect completed games to TruMedia, and move directly into pitch-level review.",
    [seasonSelect, scheduleButton],
  ));
  section.append(element("div", { className: `schedule-status ${state.schedule?.stale ? "stale" : "fresh"}`, "aria-live": "polite" }, [
    element("strong", { text: state.schedule?.games?.length ? `${state.schedule.games.length} games loaded` : "Schedule not published" }),
    element("span", { text: scheduleFreshness() }),
  ]));
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
    const runAction = gameRunAction(game, run);
    detail.append(
      element("div", { className: "game-hero" }, [
        element("div", {}, [
          element("div", { className: "context-kicker" }, [
            element("span", { className: "status-marker", "aria-hidden": "true" }),
            element("span", { text: game.conference ? "ACC matchup" : "Non-conference" }),
          ]),
          element("h2", { text: game.opponent }),
          element("p", { className: "game-line", text: `${formatDate(game.date)} · ${game.location}` }),
        ]),
        element("div", { className: "hero-facts" }, [
          element("span", { text: game.result || "Scheduled" }),
          element("small", { text: run ? `${run.result_count || 0} pitches in evidence` : "No analysis run yet" }),
        ]),
        runAction,
      ]),
      element("div", { className: "game-context-strip" }, [
        element("div", { className: "context-step" }, [
          element("span", { className: "context-label", text: "Game" }),
          element("strong", { text: game.result || "Scheduled" }),
        ]),
        element("span", { className: "context-connector", "aria-hidden": "true", text: "→" }),
        element("div", { className: "context-step" }, [
          element("span", { className: "context-label", text: "Analysis run" }),
          run ? runBadge(run) : element("strong", { text: "Not started" }),
        ]),
        element("span", { className: "context-connector", "aria-hidden": "true", text: "→" }),
        element("div", { className: "context-step" }, [
          element("span", { className: "context-label", text: "Pitch evidence" }),
          element("strong", { text: run ? `${run.result_count || 0} available` : "Not available", dataset: run ? { resultCount: run.id } : undefined }),
        ]),
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
        element("label", { className: "check-row" }, [
          element("input", { id: "retainSources", type: "checkbox", checked: true }),
          element("span", { text: "Keep full source clips available through review" }),
        ]),
        element("label", { className: "field" }, [
          element("span", { text: "Test run pitch limit" }),
          element("input", { id: "pitchLimit", type: "number", min: 1, max: 25, step: 1, placeholder: "Full game", "aria-describedby": "pitchLimitHelp" }),
        ]),
        element("p", { id: "pitchLimitHelp", className: "muted", text: "Optional. Enter 1–25 to run the real download, classification, review, and export flow on only a few pitches. Leave blank for the full game." }),
        element("p", { className: "muted", text: "Source clips remain available until every pitch is reviewed and you explicitly purge them." }),
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
  const retainSources = container.querySelector("#retainSources")?.checked !== false;
  const pitchLimitInput = container.querySelector("#pitchLimit");
  const pitchLimit = pitchLimitInput?.value ? Number(pitchLimitInput.value) : null;
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    const run = reprocess
      ? await reprocessRun(game.id, { retain_sources: retainSources, pitch_limit: pitchLimit })
      : await startRun({ game_id: game.id, retain_sources: retainSources, pitch_limit: pitchLimit });
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
  section.append(workspaceContext("Run control", `${activeRuns().length} active`));
  section.append(pageHeader(
    "Processing",
    "Run activity",
    "Track every game from media discovery through pitch-level evidence.",
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
          element("div", { className: "run-context-line" }, [
            element("span", { className: "status-marker", "aria-hidden": "true" }),
            element("span", { className: "eyebrow", text: formatDate(run.game?.date) }),
          ]),
          element("h2", { text: run.game?.opponent || run.id }),
          element("p", { className: "muted", text: run.game?.location || run.id }),
        ]),
        runBadge(run),
      ]),
      element("div", { className: "run-relationship" }, [
        element("span", { text: "Game" }),
        element("i", { "aria-hidden": "true" }),
        element("span", { text: "Run" }),
        element("i", { "aria-hidden": "true" }),
        element("span", { text: `${run.result_count || 0} pitches` }),
      ]),
      ACTIVE.has(run.status)
        ? progressBar(run)
        : element("p", { className: "run-message", text: run.message || statusLabel(run.status) }),
      element("div", { className: "run-facts" }, [
        element("span", { text: `${run.result_count || 0} results`, dataset: { resultCount: run.id } }),
        element("span", {
          text: `${(run.manifest?.downloaded || 0) + (run.manifest?.cleaned || 0)}/${run.manifest?.total || 0} clips downloaded`,
          dataset: { downloadCount: run.id },
        }),
        ...(run.revision ? [element("span", { text: `Revision ${run.revision}` })] : []),
        ...(run.cleanup?.status ? [element("span", { text: `Storage: ${statusLabel(run.cleanup.status)}` })] : []),
        element("span", { text: run.read_only ? "Read-only fixture" : "Live run" }),
      ]),
      runCardAction(run),
    ]);
    grid.append(card);
  }
  section.append(grid);
  clear(content).append(section);
}

function resultSummary(rows) {
  const accepted = rows.filter((row) => row.accepted);
  const count = (label) => rows.filter((row) => stanceKey(row.final_stance || row.stance) === label).length;
  const flagged = rows.filter(resultAttention).length;
  const reviewed = rows.filter((row) => row.review_status === "reviewed").length;
  const average = accepted.length
    ? accepted.reduce((sum, row) => sum + Number(row.confidence || 0), 0) / accepted.length
    : 0;
  return [
    ["Pitches", rows.length, "total"],
    ["Knee-down Left", count("LKD"), "LKD model label"],
    ["Knee-down Right", count("RKD"), "RKD model label"],
    ["Squat", count("Squat"), "traditional"],
    ["Needs attention", flagged, "review blockers"],
    ["Reviewed", reviewed, "final stances saved"],
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
      (state.stanceFilter === "abstained" ? !row.accepted : stanceKey(row.stance) === state.stanceFilter);
    const qualityMatch = state.qualityFilter === "all" ||
      (state.qualityFilter === "flagged" && resultAttention(row)) ||
    (state.qualityFilter === "clear" && !resultAttention(row));
    return textMatch && stanceMatch && qualityMatch;
  });
}

function resultRow(row, run) {
  const effectiveStance = row.review_status === "reviewed" ? row.final_stance : row.stance;
  const stance = stanceLabel(effectiveStance);
  const quality = row.attention_required ? "Needs attention" : row.review_status === "reviewed" ? "Reviewed" : row.accepted ? "Clear" : "Rejected";
  return element("button", {
    className: `result-row${resultAttention(row) ? " needs-attention" : ""}`,
    dataset: { clipId: row.clip_id },
    type: "button",
    on: { click: () => openPitch(run, row) },
  }, [
    element("span", { className: "pitch-number", text: String(row.pitch_index ?? "—").padStart(2, "0") }),
    element("span", { className: "stance-cell" }, [
      badge(stance, row.accepted || row.review_status === "reviewed" ? "stance" : "danger"),
      ...(row.review_status === "reviewed" && row.model_stance && row.model_stance !== row.final_stance
        ? [element("small", { text: `Model: ${stanceLabel(row.model_stance)}` })]
        : []),
    ]),
    element("span", { className: "confidence-cell" }, [
      element("strong", { text: formatPercent(row.confidence, 1) }),
      element("span", { className: "confidence-track" }, [
        element("i", { style: `width:${Math.min(100, Number(row.confidence || 0) * 100)}%` }),
      ]),
    ]),
    element("span", { className: "set-window", text: `${formatTime(row.window_start_seconds)}–${formatTime(row.window_end_seconds)}` }),
    element("span", { className: "quality-cell" }, [badge(quality, quality === "Clear" || quality === "Reviewed" ? "success" : quality === "Needs attention" ? "warning" : "danger")]),
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
  section.append(workspaceContext("Pitch evidence", run.game?.opponent || "Selected game"));
  section.append(pageHeader(
    run.read_only ? "Sample analysis" : run.results_complete ? "Completed analysis" : "Live analysis",
    `${run.game?.opponent || "Run"} results`,
    `${run.game?.opponent || "Selected game"} · ${formatDate(run.game?.date)} · ${run.result_count} pitch classifications`,
    [
      element("button", {
        className: "button sheets-button",
        type: "button",
        text: run.review?.ready ? "Preview official sheet" : `Review ${run.review?.unresolved ?? run.results?.length ?? 0} pitches to export`,
        disabled: !run.review?.ready,
        on: { click: () => openSheetPreview(run) },
      }),
      ...(run.review?.ready && !run.read_only && run.source_cleanup?.status !== "purged" ? [element("button", {
        className: "button danger-outline", type: "button", text: "Purge source videos",
        on: { click: () => purgeSourcesWithConfirmation(run) },
      })] : []),
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
  }, [["all", "All stances"], ["LKD", "Knee-down Left"], ["RKD", "Knee-down Right"], ["Squat", "Squat"], ["abstained", "Abstained"]]
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
      "Pitch", "Classification", "Confidence", "Set window", "Evidence", "",
    ].map((text) => element("span", { text }))),
    element("div", { className: "result-list" },
      rows.length ? rows.map((row) => resultRow(row, run)) : [element("div", { className: "empty-state", text: "No pitches match these filters." })]),
  ]));
  clear(content).append(section);
}

async function downloadReviewWorkbook(run) {
  try {
    const response = await fetch(apiUrl(`/api/results/${encodeURIComponent(run.id)}/xlsx`), { credentials: "same-origin" });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `Export failed (${response.status})`);
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${run.id}-catcher-review.xlsx`;
    link.click();
    URL.revokeObjectURL(link.href);
    toast("Excel review sheet downloaded.", "success");
  } catch (error) {
    toast(error.message, "danger");
  }
}

async function openSheetPreview(run) {
  const previewUrl = apiUrl(`/api/results/${encodeURIComponent(run.id)}/sheet-preview`);
  try {
    const response = await fetch(previewUrl, { credentials: "same-origin" });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const error = new Error(payload.error || `Preview failed (${response.status})`);
      Object.assign(error, payload);
      throw error;
    }
    const iframe = element("iframe", {
      className: "sheet-preview-frame",
      src: previewUrl,
      title: "Paste-ready catcher pose sheet preview",
    });
    clear(workflowDialogContent).append(
      workflowHeader("Official sheet handoff", "Preview catcher poses", "Review the exact two-column rows before downloading the Excel workbook."),
      element("div", { className: "sheet-preview-wrap" }, [iframe]),
      element("div", { className: "dialog-actions" }, [
        element("button", { className: "button ghost", type: "button", text: "Close", on: { click: closeWorkflow } }),
        element("button", { className: "button primary", type: "button", text: "Download Excel", on: { click: () => downloadReviewWorkbook(run) } }),
      ]),
    );
    workflowDialog.showModal();
  } catch (error) {
    if (error.code === "admin_required") {
      openAdminUnlock(() => openSheetPreview(run));
      return;
    }
    toast(error.message, "danger");
  }
}

function purgeSourcesWithConfirmation(run) {
  if (!window.confirm("Purge the full source clips for this reviewed game? Compact review clips, results, overrides, and exports will remain.")) return;
  openAdminUnlock(async () => {
    try {
      const response = await purgeSourceClips(run.id);
      state.runDetails.set(run.id, response.run);
      state.runs = state.runs.map((item) => item.id === run.id ? response.run : item);
      toast(`${response.purge.purged_count || 0} source clips purged.`, "success");
      closeWorkflow();
      renderResults(run.id);
    } catch (error) {
      toast(error.message, "danger");
    }
  });
}

function metadataItem(label, value) {
  return element("div", { className: "metadata-item" }, [
    element("span", { text: label }),
    element("strong", { text: value ?? "—" }),
  ]);
}

function openPitch(run, row) {
  const hasSource = row.media_mode === "source";
  const sourceUrl = apiUrl(`/api/runs/${encodeURIComponent(run.id)}/clips/${encodeURIComponent(row.clip_id)}/video?mode=source`);
  const reviewUrl = apiUrl(`/api/runs/${encodeURIComponent(run.id)}/clips/${encodeURIComponent(row.clip_id)}/video?mode=review`);
  const overlayUrl = apiUrl(`/api/runs/${encodeURIComponent(run.id)}/clips/${encodeURIComponent(row.clip_id)}/overlay.mjpg`);
  const video = element("video", { controls: "", preload: "metadata", src: row.media_mode === "source" ? sourceUrl : reviewUrl });
  const sourceDuration = element("span", { className: "media-timing-value", text: "Source duration: loading…" });
  const jumpSet = element("button", { className: "button ghost", type: "button", text: "Jump to set", disabled: true });
  const jumpEvent = element("button", { className: "button ghost", type: "button", text: "Jump to delivery", disabled: true });
  const setSeconds = Number(row.window_start_seconds);
  const eventSeconds = Number(row.impact_seconds);
  const updateTiming = () => {
    const duration = Number(video.duration);
    const valid = Number.isFinite(duration) && duration > 0;
    const outOfRange = valid && ((Number.isFinite(setSeconds) && setSeconds > duration) || (Number.isFinite(eventSeconds) && eventSeconds > duration));
    sourceDuration.textContent = valid
      ? `Source duration: ${formatTime(duration)}${outOfRange ? " · analyzer timestamp outside source" : ""}`
      : "Source duration unavailable";
    jumpSet.disabled = !valid || !Number.isFinite(setSeconds) || setSeconds < 0 || setSeconds > duration;
    jumpEvent.disabled = !valid || !Number.isFinite(eventSeconds) || eventSeconds < 0 || eventSeconds > duration;
  };
  jumpSet.addEventListener("click", () => { if (!jumpSet.disabled) video.currentTime = setSeconds; });
  jumpEvent.addEventListener("click", () => { if (!jumpEvent.disabled) video.currentTime = eventSeconds; });
  video.addEventListener("loadedmetadata", () => {
    updateTiming();
  });
  const media = element("div", { className: "pitch-media" }, [video]);
  const sourceButton = element("button", {
    className: `segmented${hasSource ? " active" : ""}`,
    type: "button",
    text: "Full pitch source",
  });
  const compactButton = element("button", { className: `segmented${hasSource ? "" : " active"}`, type: "button", text: "Compact review" });
  const reviewStatus = element("p", {
    className: "form-status",
    text: run.read_only
      ? "The checked-in sample is read-only. Run a live analysis to save review changes."
      : row.review_status === "reviewed" ? `Reviewed ${formatDateTime(row.reviewed_at)}` : "This pitch still needs a final stance.",
  });
  const reviewSelect = element("select", { "aria-label": "Final catcher pose" }, [
    ["", "Select final catcher pose"],
    ["KneeDown - Left", "KneeDown - Left"],
    ["KneeDown - Right", "KneeDown - Right"],
    ["Squat", "Squat"],
  ].map(([value, text]) => element("option", { value, text, selected: reviewStance(row.final_stance || row.stance) === value })));
  const reviewNote = element("textarea", { rows: 2, placeholder: "Optional review note", "aria-label": "Review note" });
  reviewNote.value = row.review_note || "";
  const saveReviewButton = element("button", {
    className: "button primary", type: "button", text: "Save final stance", disabled: run.read_only,
    on: { click: async () => {
      if (!reviewSelect.value) {
        reviewStatus.textContent = "Select a final catcher pose first.";
        return;
      }
      saveReviewButton.disabled = true;
      saveReviewButton.textContent = "Saving…";
      try {
        const response = await savePitchReview(run.id, row.clip_id, { stance: reviewSelect.value, note: reviewNote.value });
        Object.assign(row, response.result);
        run.review = response.summary || run.review;
        state.runDetails.set(run.id, run);
        toast("Pitch review saved.", "success");
        pitchDialog.close();
        renderResults(run.id);
      } catch (error) {
        if (error.code === "admin_required" || error.code === "admin_not_configured") {
          reviewStatus.textContent = error.code === "admin_not_configured"
            ? "Admin actions are not configured on this server. Set CATCHER_STANCE_ADMIN_TOKEN and restart it."
            : "Admin authorization is required to save this review.";
          if (error.code === "admin_required") openAdminUnlock(() => saveReviewButton.click());
          saveReviewButton.disabled = false;
          saveReviewButton.textContent = "Save final stance";
          return;
        }
        reviewStatus.textContent = error.message;
        saveReviewButton.disabled = false;
        saveReviewButton.textContent = "Save final stance";
      }
    } },
  });
  const recoverStatus = element("p", { className: "form-status", text: "The full source clip is not currently retained." });
  const recoverButton = !run.read_only && row.media_mode !== "source" ? element("button", {
    className: "button secondary", type: "button", text: "Recover full source",
    on: { click: async () => {
      recoverButton.disabled = true;
      recoverButton.textContent = "Recovering…";
      recoverStatus.textContent = "Requesting the source clip through the authenticated TruMedia session.";
      try {
        await recoverPitchSource(run.id, row.clip_id);
        for (let attempt = 0; attempt < 60; attempt += 1) {
          await new Promise((resolve) => window.setTimeout(resolve, 1000));
          const refreshed = await getRun(run.id);
          const updated = refreshed.results?.find((item) => item.clip_id === row.clip_id);
          if (updated?.media_mode === "source") {
            Object.assign(row, updated);
            video.src = sourceUrl;
            sourceButton.textContent = "Full pitch source";
            sourceButton.classList.add("active");
            compactButton.classList.remove("active");
            recoverStatus.textContent = "Full source recovered.";
            recoverButton.remove();
            return;
          }
          if (refreshed.source_recovery?.status === "failed") throw new Error(refreshed.source_recovery.error || "Source recovery failed.");
        }
        throw new Error("Source recovery is still pending. Refresh this pitch later.");
      } catch (error) {
        recoverStatus.textContent = error.message;
        recoverButton.disabled = false;
        recoverButton.textContent = "Try source recovery again";
      }
    } },
  }) : null;
  const overlayButton = element("button", {
    className: "segmented",
    type: "button",
    text: "Pose overlay",
    on: {
      click: () => {
        sourceButton.classList.remove("active");
        compactButton.classList.remove("active");
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
    compactButton.classList.remove("active");
    sourceButton.classList.add("active");
    video.src = sourceUrl;
    media.replaceChildren(video);
    video.load();
  });
  compactButton.addEventListener("click", () => {
    overlayButton.classList.remove("active");
    sourceButton.classList.remove("active");
    compactButton.classList.add("active");
    video.src = reviewUrl;
    media.replaceChildren(video);
    video.load();
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
        element("p", { className: "eyebrow", text: `${run.game?.opponent || "Game review"} · Pitch ${row.pitch_index}` }),
        element("h2", { id: "pitchDialogTitle", text: row.review_status === "reviewed" ? stanceLabel(row.final_stance) : "Needs final review" }),
        element("p", { className: "muted", text: `${formatDate(run.game?.date)} · ${row.clip_id} · Model: ${stanceLabel(row.model_stance || row.stance)}` }),
      ]),
      close,
    ]),
    element("div", { className: "media-mode" }, [sourceButton, compactButton, overlayButton]),
    media,
    element("div", { className: "media-timing" }, [sourceDuration, jumpSet, jumpEvent]),
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
      element("section", { className: "detail-section review-editor" }, [
        element("h3", { text: "Final review" }),
        element("label", { className: "field" }, [element("span", { text: "Catcher Pose" }), reviewSelect]),
        element("label", { className: "field" }, [element("span", { text: "Review note" }), reviewNote]),
        reviewStatus,
        element("div", { className: "dialog-actions" }, [saveReviewButton]),
      ]),
      element("section", { className: "detail-section" }, [
        element("h3", { text: "Quality and provenance" }),
        element("div", { className: "tag-list" }, [
          ...(row.quality_flags?.length ? row.quality_flags.map((flag) => badge(statusLabel(flag), "warning")) : [badge("No quality flags", "success")]),
          ...(row.detector_provenance || []).map((item) => badge(statusLabel(item), "neutral")),
        ]),
      ]),
      ...(recoverButton ? [element("section", { className: "detail-section" }, [element("h3", { text: "Full source" }), recoverStatus, recoverButton])] : []),
      element("section", { className: "detail-section" }, [
        element("h3", { text: "Review action" }),
        element("p", { className: "muted", text: resultAttention(row) ? "This pitch needs analyst attention before it is treated as a clear classification." : "This pitch is ready for review or export." }),
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
  document.querySelectorAll(`[data-download-count="${run.id}"]`).forEach((node) => {
    node.textContent = `${(run.manifest?.downloaded || 0) + (run.manifest?.cleaned || 0)}/${run.manifest?.total || 0} clips downloaded`;
  });
  document.querySelectorAll(`[data-run-action="${run.id}"]`).forEach((node) => {
    if (node.dataset.actionKind === "card") {
      const replacement = runCardAction(run);
      if (node.dataset.actionState !== replacement.dataset.actionState) node.replaceWith(replacement);
    }
    if (node.dataset.actionKind === "game") {
      const game = state.schedule?.games?.find((item) => item.id === run.game?.id);
      if (game) {
        const replacement = gameRunAction(game, run);
        if (node.dataset.actionState !== replacement.dataset.actionState) node.replaceWith(replacement);
      }
    }
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
    if (summary && (!old || summary.result_count !== old.result_count)) {
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
    const [scheduleSettings, seasonsPayload, trumedia, summaries] = await Promise.all([
      getScheduleSettings(),
      getSeasons(),
      getTruMediaStatus(),
      getRunSummaries(),
    ]);
    state.scheduleSettings = scheduleSettings;
    state.activeSeason = scheduleSettings.active_season;
    state.season = scheduleSettings.active_season;
    state.schedule = await loadScheduleForSeason(state.season);
    state.seasons = seasonsPayload.seasons || [state.season];
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
