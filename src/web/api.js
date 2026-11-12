const config = window.CATCHER_APP_CONFIG || {};
const API_BASE = (config.apiBaseUrl || window.location.origin).replace(/\/$/, "");

export function apiUrl(path) {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function request(path, options = {}) {
  const isForm = options.body instanceof FormData;
  const response = await fetch(apiUrl(path), {
    headers: { ...(isForm ? {} : { "Content-Type": "application/json" }), ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    Object.assign(error, payload);
    throw error;
  }
  return payload;
}

export const getSchedule = (season = 2026) => request(`/api/schedule?season=${encodeURIComponent(season)}&include_scheduled=1`);
export const getSeasons = () => request("/api/teams/duke/seasons");
export const getScheduleSettings = () => request("/api/teams/duke/schedule-settings");
export const setActiveSeason = (season) =>
  request("/api/teams/duke/schedule-settings", {
    method: "PUT",
    body: JSON.stringify({ active_season: Number(season) }),
  });
export const refreshSchedule = (season) =>
  request("/api/teams/duke/schedule/refresh", {
    method: "POST",
    body: JSON.stringify({ season: Number(season) }),
  });
export const getRuns = () => request("/api/runs");
export async function getRunSummaries(etag = null, signal = undefined) {
  const response = await fetch(apiUrl("/api/runs?view=summary"), {
    headers: etag ? { "If-None-Match": etag } : {},
    signal,
  });
  if (response.status === 304) return { notModified: true, etag };
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    Object.assign(error, payload);
    throw error;
  }
  return { payload, etag: response.headers.get("ETag") };
}
export const getTruMediaStatus = () => request("/api/integrations/trumedia/status");
export const getRun = (runId, options = {}) => request(`/api/runs/${encodeURIComponent(runId)}`, options);
export const controlRun = (runId, action) =>
  request(`/api/runs/${encodeURIComponent(runId)}/control`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });
export const deleteRun = (runId) =>
  request(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
export const savePitchReview = (runId, clipId, payload) =>
  request(`/api/runs/${encodeURIComponent(runId)}/clips/${encodeURIComponent(clipId)}/review`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
export const recoverPitchSource = (runId, clipId) =>
  request(`/api/runs/${encodeURIComponent(runId)}/clips/${encodeURIComponent(clipId)}/source-recovery`, {
    method: "POST",
  });
export const startRun = (payload) =>
  request("/api/run", { method: "POST", body: JSON.stringify(payload) });
export const reprocessRun = (gameId, payload) =>
  request(`/api/games/${encodeURIComponent(gameId)}/reprocess`, { method: "POST", body: JSON.stringify(payload) });
export const unlockAdmin = (token) =>
  request("/api/admin/session", { method: "POST", body: JSON.stringify({ token }) });
export const uploadTruMediaSession = (file) => {
  const body = new FormData();
  body.append("session", file);
  return request("/api/integrations/trumedia/session", { method: "POST", body });
};
export const validateTruMediaSession = () =>
  request("/api/integrations/trumedia/validate", { method: "POST" });
export const confirmTruMediaMatch = (gameId, candidateId) =>
  request(`/api/games/${encodeURIComponent(gameId)}/trumedia-match`, {
    method: "POST",
    body: JSON.stringify({ candidate_id: candidateId }),
  });
