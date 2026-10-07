// src/components/FlaggedPanel.jsx

function SeverityBadge({ severity }) {
  const map = {
    high:   { cls: 'badge-error',   label: 'High' },
    medium: { cls: 'badge-warning', label: 'Medium' },
    low:    { cls: 'badge-blue',    label: 'Low' },
  };
  const cfg = map[severity?.toLowerCase()] || map.medium;
  return <span className={`badge ${cfg.cls}`}>{cfg.label} Risk</span>;
}

export default function FlaggedPanel({ flags = [], similarityWarning = null }) {
  const hasContent = flags.length > 0 || similarityWarning;

  return (
    <section aria-label="Flagged statements and verification">
      <h3 style={{ marginBottom: 'var(--space-4)' }}>Verification Flags</h3>

      {/* Similarity warning banner */}
      {similarityWarning && (
        <div className="similarity-warning" role="alert">
          <span aria-hidden="true" style={{ fontSize: '1.3rem', flexShrink: 0 }}>⚡</span>
          <div>
            <strong>Low Similarity Score Detected</strong>
            <p>{similarityWarning}</p>
          </div>
        </div>
      )}

      {/* Flags list */}
      {flags.length > 0 ? (
        <div role="list" aria-label="Flagged claims">
          {flags.map((f, idx) => (
            <div
              key={idx}
              className={`flag-item severity-${(f.severity || 'medium').toLowerCase()}`}
              role="listitem"
            >
              <div className="flag-header">
                <span className="flag-claim">🚩 {f.claim || f.statement || `Flag #${idx + 1}`}</span>
                <SeverityBadge severity={f.severity} />
              </div>
              {f.evidence && (
                <div className="flag-evidence">
                  <strong style={{ color: 'var(--clr-text-secondary)' }}>Evidence: </strong>
                  {f.evidence}
                </div>
              )}
              {f.source && (
                <div className="flag-evidence" style={{ marginTop: 4 }}>
                  <strong style={{ color: 'var(--clr-text-secondary)' }}>Source: </strong>
                  {f.source}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        !similarityWarning && (
          <div className="no-flags" role="status">
            <div className="no-flags-icon" aria-hidden="true">✅</div>
            <p>No statements were flagged as unsupported or inconsistent.</p>
          </div>
        )
      )}

      {!hasContent && null}
    </section>
  );
}
