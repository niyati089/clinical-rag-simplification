// src/api.js
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const resp = await fetch(`${BASE}${path}`, options);
  if (!resp.ok) {
    let msg = `Server error ${resp.status}`;
    try {
      const data = await resp.json();
      msg = data.detail || data.message || msg;
    } catch (_) { /* non-json body */ }
    throw new Error(msg);
  }
  return resp.json();
}

export function getHealth() {
  return request('/health');
}

export function uploadDocument(file) {
  const form = new FormData();
  form.append('file', file);
  return request('/api/documents', { method: 'POST', body: form });
}

export function startSimplify({ text, readingLevel }) {
  return request('/api/simplify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, reading_level: readingLevel }),
  });
}

export function getJob(jobId) {
  return request(`/api/jobs/${jobId}`);
}
