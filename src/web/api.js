const config = window.CATCHER_APP_CONFIG || {};
const API_BASE = (config.apiBaseUrl || window.location.origin).replace(/\/$/, "");

export function apiUrl(path) {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function request(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

export const getSchedule = () => request("/api/schedule");
export const getRuns = () => request("/api/runs");
export const getRun = (runId) => request(`/api/runs/${encodeURIComponent(runId)}`);
export const startRun = (payload) =>
  request("/api/run", { method: "POST", body: JSON.stringify(payload) });
