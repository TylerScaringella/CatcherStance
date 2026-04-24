// Codex attribution: OpenAI Codex generated the browser-side web interface
// logic, with project-specific review and edits.
const state = {
  schedule: null,
  games: [],
  selectedGame: null,
  activeJobId: null,
  pollTimer: null,
  gameStatus: {},
};

const config = window.CATCHER_APP_CONFIG || {};
const API_BASE_URL = (config.apiBaseUrl || window.location.origin).replace(/\/$/, "");

const els = {
  scheduleMeta: document.querySelector("#scheduleMeta"),
  gameList: document.querySelector("#gameList"),
  gameTitle: document.querySelector("#gameTitle"),
  gameDetails: document.querySelector("#gameDetails"),
  trumediaUrl: document.querySelector("#trumediaUrl"),
  runBtn: document.querySelector("#runBtn"),
  jobStatus: document.querySelector("#jobStatus"),
  jobMessage: document.querySelector("#jobMessage"),
  resultsBody: document.querySelector("#resultsBody"),
  mobileResults: document.querySelector("#mobileResults"),
  jsonExport: document.querySelector("#jsonExport"),
  csvExport: document.querySelector("#csvExport"),
  search: document.querySelector("#search"),
  statusFilter: document.querySelector("#statusFilter"),
  forceRedownload: document.querySelector("#forceRedownload"),
  progressBar: document.querySelector("#progressBar"),
  replayDialog: document.querySelector("#replayDialog"),
  replayTitle: document.querySelector("#replayTitle"),
  replayMeta: document.querySelector("#replayMeta"),
  overlayStream: document.querySelector("#overlayStream"),
  closeReplay: document.querySelector("#closeReplay"),
};

function apiUrl(path) {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

function formatDate(value) {
  const date = new Date(`${value}T12:00:00`);
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" }).format(date);
}

function isComplete(game) {
  return /^[WL]\s/.test(game.result || "");
}

function filteredGames() {
  const query = els.search.value.trim().toLowerCase();
  const filter = els.statusFilter.value;
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  return state.games.filter((game) => {
    const searchable = `${game.opponent} ${game.location}`.toLowerCase();
    const matchesQuery = !query || searchable.includes(query);
    const gameDate = new Date(`${game.date}T00:00:00`);
    const matchesFilter =
      filter === "all" ||
      (filter === "complete" && isComplete(game)) ||
      (filter === "future" && gameDate >= today && !isComplete(game)) ||
      (filter === "conference" && game.conference);
    return matchesQuery && matchesFilter;
  }).sort((a, b) => {
    const dateCompare = b.date.localeCompare(a.date);
    if (dateCompare !== 0) return dateCompare;
    return b.id.localeCompare(a.id);
  });
}

function renderGames() {
  const games = filteredGames();
  els.gameList.innerHTML = "";

  for (const game of games) {
    const status = state.gameStatus[game.id] || { status: "none", label: "Not run" };
    const badge = statusBadge(status);
    const button = document.createElement("button");
    button.className = `game-card${state.selectedGame?.id === game.id ? " active" : ""}`;
    button.innerHTML = `
      <div class="date">${formatDate(game.date)}</div>
      <div>
        <div class="game-title-row">
          <div class="opponent">${game.opponent}</div>
          ${badge}
        </div>
        <div class="meta">${game.location} · ${game.result || "Scheduled"}${game.conference ? " · ACC" : ""}</div>
      </div>
    `;
    button.addEventListener("click", () => selectGame(game));
    els.gameList.appendChild(button);
  }
}

function statusBadge(status) {
  const variants = {
    complete: { icon: "✓", label: status.label || "Detected", title: "Detection complete" },
    running: { icon: "…", label: status.label || "In progress", title: "Run in progress" },
    queued: { icon: "…", label: status.label || "Queued", title: "Run queued" },
    ready: { icon: "↓", label: status.label || "Downloaded", title: "Videos downloaded; detection not run" },
    interrupted: { icon: "!", label: status.label || "Partial", title: "Partial or interrupted run" },
    failed: { icon: "!", label: status.label || "Failed", title: "Run failed" },
  };
  const variant = variants[status.status];
  if (!variant) return "";
  return `
    <span class="status-badge ${status.status}" title="${variant.title}">
      <span class="status-icon">${variant.icon}</span>
      <span>${variant.label}</span>
    </span>
  `;
}

async function selectGame(game) {
  state.selectedGame = game;
  els.gameTitle.textContent = `${formatDate(game.date)} vs ${game.opponent.replace(/^at /, "")}`;
  els.gameDetails.textContent = `${game.location} · ${game.result || "Scheduled"}${game.conference ? " · ACC" : ""}`;
  els.trumediaUrl.value = game.trumedia_url || "https://duke-ncaabaseball.trumedianetworks.com/baseball/";
  els.runBtn.disabled = false;
  setExports("", false);
  renderResults([]);
  setJob("Idle", "Checking for existing runs");
  renderGames();
  await loadLatestJob(game.id);
}

function setJob(status, message, progress = null) {
  els.jobStatus.textContent = status;
  els.jobMessage.textContent = message || "";
  const percent = progress?.percent ?? 0;
  els.progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  els.progressBar.parentElement.classList.toggle("visible", Boolean(progress));
}

function setExports(jobId, enabled) {
  for (const [el, fmt] of [[els.jsonExport, "json"], [els.csvExport, "csv"]]) {
    el.href = enabled ? apiUrl(`/api/results/${jobId}/${fmt}`) : "#";
    el.classList.toggle("disabled", !enabled);
  }
}

function renderResults(rows) {
  if (!rows || rows.length === 0) {
    els.resultsBody.innerHTML = '<tr><td colspan="6" class="empty">No detections yet.</td></tr>';
    els.mobileResults.innerHTML = '<div class="empty">No detections yet.</div>';
    return;
  }

  els.resultsBody.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.pitch_index}</td>
      <td title="${row.clip_id}">${row.clip_id}</td>
      <td>${row.stance || ""}</td>
      <td>${row.confidence ? `${(row.confidence * 100).toFixed(1)}%` : ""}</td>
      <td>${row.status}</td>
      <td>
        <button class="icon-action" type="button" data-clip-id="${row.clip_id}" data-stance="${row.stance || ""}" data-confidence="${row.confidence || ""}">
          Play
        </button>
      </td>
    </tr>
  `).join("");

  els.resultsBody.querySelectorAll(".icon-action").forEach((button) => {
    button.addEventListener("click", () => openReplay(button.dataset));
  });

  els.mobileResults.innerHTML = rows.map((row) => `
    <button class="pitch-card" type="button" data-clip-id="${row.clip_id}" data-stance="${row.stance || ""}" data-confidence="${row.confidence || ""}">
      <span class="pitch-card-main">
        <span>Pitch ${row.pitch_index}</span>
        <strong>${row.stance || row.status}</strong>
      </span>
      <span class="pitch-card-sub">${row.confidence ? `${(row.confidence * 100).toFixed(1)}%` : row.status}</span>
    </button>
  `).join("");

  els.mobileResults.querySelectorAll(".pitch-card").forEach((button) => {
    button.addEventListener("click", () => openReplay(button.dataset));
  });
}

function openReplay(data) {
  if (!state.activeJobId || !data.clipId) return;

  const confidence = data.confidence ? ` · ${(Number(data.confidence) * 100).toFixed(1)}%` : "";
  els.replayTitle.textContent = data.clipId;
  els.replayMeta.textContent = `${data.stance || "Pitch-level stance unavailable"}${confidence}`;
  els.overlayStream.src = apiUrl(`/api/jobs/${encodeURIComponent(state.activeJobId)}/clips/${encodeURIComponent(data.clipId)}/overlay.mjpg?ts=${Date.now()}`);
  els.replayDialog.showModal();
}

function closeReplay() {
  els.overlayStream.removeAttribute("src");
  els.replayDialog.close();
}

function applyJob(job) {
  state.activeJobId = job.id;
  if (job.game?.id) {
    state.gameStatus[job.game.id] = {
      status: job.status,
      label: job.status === "complete" ? `Detected ${job.result_count || 0}` : job.message,
      job_id: job.id,
      result_count: job.result_count || 0,
      manifest: job.manifest,
      progress: job.progress,
    };
    renderGames();
  }
  setJob(job.status, job.message, job.progress);
  renderResults(job.results);
  setExports(job.id, job.status === "complete");
  els.runBtn.disabled = job.status === "queued" || job.status === "running";

  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }

  if (job.status === "queued" || job.status === "running") {
    state.pollTimer = setInterval(() => pollJob(job.id), 2500);
  }
}

async function loadLatestJob(gameId) {
  const res = await fetch(apiUrl(`/api/games/${gameId}/latest-job`));
  if (state.selectedGame?.id !== gameId) return;

  if (res.status === 404) {
    setJob("Idle", "No run has been started for this game");
    return;
  }

  const job = await res.json();
  if (res.ok) {
    applyJob(job);
  } else {
    setJob("Idle", job.error || "Unable to load run status");
  }
}

async function pollJob(jobId) {
  const res = await fetch(apiUrl(`/api/jobs/${jobId}`));
  const job = await res.json();
  if (!res.ok) {
    setJob("Idle", job.error || "Unable to load run status");
    return;
  }
  applyJob(job);
}

async function runDetection() {
  if (!state.selectedGame) return;

  els.runBtn.disabled = true;
  setExports("", false);
  renderResults([]);
  setJob("queued", "Starting detection job");

  const res = await fetch(apiUrl("/api/run"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      game_id: state.selectedGame.id,
      trumedia_url: els.trumediaUrl.value.trim(),
      force_redownload: els.forceRedownload.checked,
    }),
  });
  const job = await res.json();
  if (!res.ok) {
    setJob("failed", job.error || "Unable to start job");
    els.runBtn.disabled = false;
    return;
  }

  state.activeJobId = job.id;
  applyJob(job);
}

async function init() {
  const [scheduleRes, statusRes] = await Promise.all([
    fetch(apiUrl("/api/schedule")),
    fetch(apiUrl("/api/game-status")),
  ]);
  state.schedule = await scheduleRes.json();
  state.gameStatus = await statusRes.json();
  state.games = [...state.schedule.games].sort((a, b) => {
    const dateCompare = b.date.localeCompare(a.date);
    if (dateCompare !== 0) return dateCompare;
    return b.id.localeCompare(a.id);
  });
  els.scheduleMeta.textContent = `${state.schedule.team} ${state.schedule.season} ${state.schedule.sport} · source checked ${state.schedule.source_checked}`;
  renderGames();
}

els.runBtn.addEventListener("click", runDetection);
els.search.addEventListener("input", renderGames);
els.statusFilter.addEventListener("change", renderGames);
els.closeReplay.addEventListener("click", closeReplay);
els.replayDialog.addEventListener("close", () => {
  els.overlayStream.removeAttribute("src");
});

init();
