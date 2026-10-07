// src/components/ResultsView.jsx
import SideBySide from './SideBySide.jsx';
import SimplifiedSections from './SimplifiedSections.jsx';
import GlossaryPanel from './GlossaryPanel.jsx';
import FlaggedPanel from './FlaggedPanel.jsx';
import ScoresPanel from './ScoresPanel.jsx';
import ActionBar from './ActionBar.jsx';

export default function ResultsView({ originalText = '', result = {}, onReset }) {
  // Defensively extract fields — handle varying backend response shapes
  const simplifiedText =
    result.simplified_text ||
    result.sections?.simple_explanation ||
    '';

  const sections      = result.sections      || {};
  const glossary      = result.glossary      || {};
  const flags         = result.flags         || result.flagged_statements || [];
  const similarityWarn = result.similarity_warning || (
    typeof result.similarity === 'number' && result.similarity < 0.6
      ? `Overall semantic similarity is ${Math.round(result.similarity * 100)}%, which is below the recommended threshold of 60%. Please review the simplified output carefully.`
      : null
  );
  const scores = result.scores || {
    flesch_before: result.flesch_before ?? null,
    flesch_after:  result.flesch_after  ?? null,
    grade_before:  result.grade_before  ?? null,
    grade_after:   result.grade_after   ?? null,
    similarity:    result.similarity    ?? null,
  };

  return (
    <div className="results-view slide-in">
      {/* ── Top bar ── */}
      <div className="results-header">
        <h2>Simplification Results</h2>
        <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
          {onReset && (
            <button
              id="btn-start-over"
              className="btn btn-ghost"
              onClick={onReset}
              aria-label="Start over with a new document"
            >
              ← Start Over
            </button>
          )}
        </div>
      </div>

      <div className="results-grid">
        {/* 1. Side-by-side comparison */}
        <div className="card">
          <div className="card-body">
            <SideBySide originalText={originalText} simplifiedText={simplifiedText} />
          </div>
        </div>

        {/* 2. Structured simplified sections */}
        <div className="card">
          <div className="card-body">
            <SimplifiedSections sections={sections} />
          </div>
        </div>

        {/* 3. Glossary with inline term tooltips */}
        <div className="card">
          <div className="card-body">
            <GlossaryPanel text={simplifiedText} glossary={glossary} />
          </div>
        </div>

        {/* 4. Flagged statements + similarity warning */}
        <div className="card">
          <div className="card-body">
            <FlaggedPanel flags={flags} similarityWarning={similarityWarn} />
          </div>
        </div>

        {/* 5. Scores */}
        <div className="card">
          <div className="card-body">
            <ScoresPanel scores={scores} />
          </div>
        </div>

        {/* 6. Action bar (sticky) */}
        <ActionBar result={result} />
      </div>
    </div>
  );
}
