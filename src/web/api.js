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

export const getSchedule = (season = 2026) => request(`/api/schedule?season=${encodeURIComponent(season)}`);
export const getSeasons = () => request("/api/teams/duke/seasons");
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
