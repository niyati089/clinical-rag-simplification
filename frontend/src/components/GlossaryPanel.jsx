// src/components/GlossaryPanel.jsx
import { useState, useRef } from 'react';

// Renders text with clickable/hoverable medical term spans
function AnnotatedText({ text = '', glossary = {} }) {
  const [activeterm, setActiveTerm] = useState(null);
  const tooltipRef = useRef(null);

  if (!text || Object.keys(glossary).length === 0) {
    return (
      <div className="glossary-text">
        {text || <span className="text-muted">No text available.</span>}
      </div>
    );
  }

  // Build a regex that matches any term (case-insensitive, whole word preferred)
  const terms = Object.keys(glossary);
  const escaped = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const regex = new RegExp(`\\b(${escaped.join('|')})\\b`, 'gi');

  // Split text into plain-string and term parts
  const parts = [];
  let lastIdx = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push({ type: 'text', value: text.slice(lastIdx, match.index) });
    }
    parts.push({ type: 'term', value: match[0], key: match[0].toLowerCase() });
    lastIdx = regex.lastIndex;
  }
  if (lastIdx < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIdx) });
  }

  return (
    <div className="glossary-text">
      {parts.map((part, i) => {
        if (part.type === 'text') return <span key={i}>{part.value}</span>;
        const def = glossary[part.key] || glossary[part.value] || '(definition not available)';
        const isOpen = activeterm === `${part.value}-${i}`;
        return (
          <span
            key={i}
            className="tooltip-wrapper"
          >
            <span
              className="tooltip-trigger"
              tabIndex={0}
              role="button"
              aria-haspopup="dialog"
              aria-expanded={isOpen}
              aria-label={`${part.value} — click for definition`}
              onClick={() => setActiveTerm(isOpen ? null : `${part.value}-${i}`)}
              onMouseEnter={() => setActiveTerm(`${part.value}-${i}`)}
              onMouseLeave={() => setActiveTerm(null)}
              onFocus={() => setActiveTerm(`${part.value}-${i}`)}
              onBlur={() => setActiveTerm(null)}
              onKeyDown={e => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setActiveTerm(isOpen ? null : `${part.value}-${i}`);
                }
                if (e.key === 'Escape') setActiveTerm(null);
              }}
            >
              {part.value}
            </span>
            {isOpen && (
              <div
                ref={tooltipRef}
                className="tooltip-box"
                role="dialog"
                aria-label={`Definition: ${part.value}`}
              >
                <strong>{part.value}</strong>
                {def}
              </div>
            )}
          </span>
        );
      })}
    </div>
  );
}

export default function GlossaryPanel({ text = '', glossary = {} }) {
  const terms = Object.entries(glossary);

  return (
    <section aria-label="Medical terms glossary">
      <h3 style={{ marginBottom: 'var(--space-2)' }}>Medical Term Glossary</h3>
      <p className="glossary-intro">
        Hover or click <span className="tooltip-trigger" style={{ cursor: 'default' }}>highlighted terms</span> in the text below to see their plain-language definition.
      </p>

      {terms.length === 0 && !text ? (
        <div className="empty-state">
          <div className="empty-state-icon" aria-hidden="true">🔬</div>
          <p>No medical terms or simplified text to display.</p>
        </div>
      ) : (
        <>
          <AnnotatedText text={text} glossary={glossary} />

          {terms.length > 0 && (
            <>
              <h4 style={{ marginTop: 'var(--space-6)', marginBottom: 'var(--space-4)' }}>
                All Defined Terms ({terms.length})
              </h4>
              <div className="glossary-terms-list">
                {terms.map(([term, definition]) => (
                  <div key={term} className="glossary-term-card">
                    <strong>{term}</strong>
                    <p>{definition}</p>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
