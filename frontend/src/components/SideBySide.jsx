// src/components/SideBySide.jsx

function countWords(text = '') {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

export default function SideBySide({ originalText = '', simplifiedText = '' }) {
  const origWords = countWords(originalText);
  const simpWords = countWords(simplifiedText);
  const reduction = origWords > 0 ? Math.round((1 - simpWords / origWords) * 100) : 0;

  return (
    <section aria-label="Side-by-side comparison">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-4)', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
        <h3 style={{ margin: 0 }}>Document Comparison</h3>
        {reduction > 0 && (
          <span className="badge badge-teal">
            ↓ {reduction}% shorter
          </span>
        )}
      </div>

      <div className="sidebyside-grid">
        {/* Original */}
        <div className="card sidebyside-col">
          <div className="card-header" style={{ justifyContent: 'space-between' }}>
            <h4 style={{ margin: 0 }}>Original</h4>
            <span className="badge badge-muted">{origWords} words</span>
          </div>
          {originalText ? (
            <div className="sidebyside-text" tabIndex={0} aria-label="Original document text">
              {originalText}
            </div>
          ) : (
            <div className="empty-state" style={{ padding: 'var(--space-8)' }}>
              <p>No original text available.</p>
            </div>
          )}
        </div>

        {/* Simplified */}
        <div className="card sidebyside-col">
          <div className="card-header" style={{ justifyContent: 'space-between' }}>
            <h4 style={{ margin: 0 }}>Simplified</h4>
            <span className="badge badge-success">{simpWords} words</span>
          </div>
          {simplifiedText ? (
            <div className="sidebyside-text" tabIndex={0} aria-label="Simplified document text">
              {simplifiedText}
            </div>
          ) : (
            <div className="empty-state" style={{ padding: 'var(--space-8)' }}>
              <p>Simplified text not yet available.</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
