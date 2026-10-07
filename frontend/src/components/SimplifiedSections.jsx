// src/components/SimplifiedSections.jsx
import { useState } from 'react';

const SECTION_CONFIG = [
  {
    key: 'simple_explanation',
    label: 'Simple Explanation',
    icon: '💬',
    ariaLabel: 'Simple explanation of the document',
    highlighted: false,
  },
  {
    key: 'important_instructions',
    label: 'Important Instructions',
    icon: '⚠️',
    ariaLabel: 'Important instructions to follow',
    highlighted: true,
  },
  {
    key: 'medication_guidance',
    label: 'Medication Guidance',
    icon: '💊',
    ariaLabel: 'Medication guidance from the document',
    highlighted: false,
  },
  {
    key: 'follow_up',
    label: 'Follow-up Steps',
    icon: '📅',
    ariaLabel: 'Follow-up steps and appointments',
    highlighted: false,
  },
];

function SectionCard({ config, content }) {
  const [open, setOpen] = useState(true);
  const isEmpty = !content || content.trim().length === 0;

  return (
    <article
      className={`section-card ${config.highlighted ? 'highlighted' : ''}`}
      aria-label={config.ariaLabel}
    >
      <div
        className="section-card-header"
        onClick={() => setOpen(o => !o)}
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && setOpen(o => !o)}
      >
        <span className="section-icon" aria-hidden="true">{config.icon}</span>
        <span className="section-title">{config.label}</span>
        {config.highlighted && (
          <span className="badge badge-warning" style={{ marginRight: 8 }}>Action Required</span>
        )}
        <span className={`section-chevron ${open ? 'open' : ''}`} aria-hidden="true">▾</span>
      </div>

      {open && (
        <div className="section-body" id={`section-body-${config.key}`}>
          {isEmpty ? (
            <div className="section-empty">No {config.label.toLowerCase()} in this document.</div>
          ) : (
            <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{content}</p>
          )}
        </div>
      )}
    </article>
  );
}

export default function SimplifiedSections({ sections = {} }) {
  const hasAny = SECTION_CONFIG.some(c => sections[c.key] && sections[c.key].trim().length > 0);

  return (
    <section aria-label="Simplified content sections">
      <h3 style={{ marginBottom: 'var(--space-4)' }}>Simplified Sections</h3>

      {!hasAny && (
        <div className="empty-state">
          <div className="empty-state-icon" aria-hidden="true">📋</div>
          <p>No structured sections were returned. Check the side-by-side view for simplified text.</p>
        </div>
      )}

      <div className="sections-grid">
        {SECTION_CONFIG.map(cfg => (
          <SectionCard key={cfg.key} config={cfg} content={sections[cfg.key] || ''} />
        ))}
      </div>
    </section>
  );
}
