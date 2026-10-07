// src/components/ScoresPanel.jsx

function scoreClass(value, max = 100) {
  const pct = value / max;
  if (pct >= 0.7) return 'high';
  if (pct >= 0.45) return 'mid';
  return 'low';
}

function similarityClass(value) {
  if (value >= 0.8) return 'high';
  if (value >= 0.6) return 'mid';
  return 'low';
}

function fkDescription(score) {
  if (score === null || score === undefined) return 'N/A';
  if (score >= 90) return 'Very Easy';
  if (score >= 70) return 'Easy';
  if (score >= 50) return 'Standard';
  if (score >= 30) return 'Difficult';
  return 'Very Difficult';
}

function StatCard({ label, value, sub, colorCls }) {
  const display = value !== null && value !== undefined ? (typeof value === 'number' ? value.toFixed(1) : value) : '—';
  return (
    <div className="score-stat">
      <div className="score-label">{label}</div>
      <div className={`score-value ${colorCls}`}>{display}</div>
      {sub && <div className="score-sub">{sub}</div>}
    </div>
  );
}

export default function ScoresPanel({ scores = {} }) {
  const {
    flesch_before = null,
    flesch_after  = null,
    grade_before  = null,
    grade_after   = null,
    similarity    = null,
  } = scores;

  const simPct = similarity !== null ? Math.round(similarity * 100) : null;

  return (
    <section aria-label="Readability and similarity scores">
      <h3 style={{ marginBottom: 'var(--space-5)' }}>Readability Scores</h3>

      <div className="scores-grid" style={{ marginBottom: 'var(--space-6)' }}>
        <StatCard
          label="Flesch Reading Ease (Before)"
          value={flesch_before}
          sub={fkDescription(flesch_before)}
          colorCls={flesch_before !== null ? scoreClass(flesch_before) : ''}
        />
        <StatCard
          label="Flesch Reading Ease (After)"
          value={flesch_after}
          sub={fkDescription(flesch_after)}
          colorCls={flesch_after !== null ? scoreClass(flesch_after) : ''}
        />
        <StatCard
          label="Grade Level (Before)"
          value={grade_before}
          sub="Flesch–Kincaid"
          colorCls={grade_before !== null ? (grade_before <= 8 ? 'high' : grade_before <= 12 ? 'mid' : 'low') : ''}
        />
        <StatCard
          label="Grade Level (After)"
          value={grade_after}
          sub="Flesch–Kincaid"
          colorCls={grade_after !== null ? (grade_after <= 8 ? 'high' : grade_after <= 12 ? 'mid' : 'low') : ''}
        />
      </div>

      {/* Similarity score */}
      <h3 style={{ marginBottom: 'var(--space-4)' }}>Semantic Similarity</h3>
      {similarity !== null ? (
        <div style={{ maxWidth: 480 }}>
          <div className="similarity-bar-label">
            <span style={{ color: 'var(--clr-text-secondary)', fontSize: '0.88rem' }}>
              How faithfully the simplification preserves meaning
            </span>
            <strong className={`text-${similarityClass(similarity) === 'high' ? 'success' : similarityClass(similarity) === 'mid' ? 'warning' : 'error'}`}>
              {simPct}%
            </strong>
          </div>
          <div className="progress-bar-track" role="progressbar" aria-valuenow={simPct} aria-valuemin={0} aria-valuemax={100}>
            <div
              className={`progress-bar-fill ${similarityClass(similarity)}`}
              style={{ width: `${simPct}%` }}
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: '0.75rem', color: 'var(--clr-text-muted)' }}>
            <span>Low fidelity</span>
            <span>High fidelity</span>
          </div>
          {similarity < 0.6 && (
            <p style={{ marginTop: 'var(--space-3)', color: 'var(--clr-warning)', fontSize: '0.85rem' }}>
              ⚠ Similarity score below 60% — the simplified version may not accurately represent the original.
            </p>
          )}
        </div>
      ) : (
        <div className="empty-state" style={{ padding: 'var(--space-6)' }}>
          <p>Similarity score not available.</p>
        </div>
      )}
    </section>
  );
}
