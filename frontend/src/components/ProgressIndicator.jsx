// src/components/ProgressIndicator.jsx
import { useState, useEffect } from 'react';

const STAGES = [
  { key: 'queued',      label: 'Job queued',             icon: '⏳' },
  { key: 'extracting', label: 'Extracting text',         icon: '📄' },
  { key: 'chunking',   label: 'Chunking document',       icon: '✂️' },
  { key: 'embedding',  label: 'Building embeddings',     icon: '🔗' },
  { key: 'simplifying',label: 'Simplifying content',     icon: '✨' },
  { key: 'verifying',  label: 'Verifying accuracy',      icon: '🔍' },
  { key: 'done',        label: 'Complete',               icon: '✅' },
];

function getStageIndex(stage) {
  const idx = STAGES.findIndex(s => s.key === stage);
  return idx === -1 ? 0 : idx;
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default function ProgressIndicator({ stage, readingLevel }) {
  const [elapsed, setElapsed] = useState(0);
  const activeIdx = getStageIndex(stage);

  useEffect(() => {
    const timer = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const currentStage = STAGES[activeIdx] || STAGES[0];

  return (
    <div className="progress-screen fade-in" role="status" aria-live="polite" aria-label="Processing status">
      <div className="progress-spinner-wrapper">
        <div className="progress-spinner" aria-hidden="true" />
      </div>

      <h2>Simplifying your document</h2>
      <p>
        Target level: <strong style={{ color: 'var(--clr-accent)', textTransform: 'capitalize' }}>{readingLevel}</strong>
        {' · '}Currently: <span style={{ color: 'var(--clr-text-primary)' }}>{currentStage.label}</span>
      </p>

      <div className="progress-steps" aria-label="Processing stages">
        {STAGES.filter(s => s.key !== 'done').map((s, idx) => {
          const isDone = idx < activeIdx;
          const isActive = idx === activeIdx;
          const isPending = idx > activeIdx;
          return (
            <div
              key={s.key}
              className={`progress-step ${isDone ? 'done' : isActive ? 'active' : 'pending'}`}
              aria-current={isActive ? 'step' : undefined}
            >
              {isActive
                ? <span className="step-active-spinner" aria-hidden="true" />
                : <span className="step-icon" aria-hidden="true">{isDone ? '✓' : isPending ? '○' : s.icon}</span>
              }
              <span>{s.label}</span>
              {isDone && <span className="text-xs text-muted" style={{ marginLeft: 'auto' }}>Done</span>}
            </div>
          );
        })}
      </div>

      <p className="progress-timer" aria-live="off">Elapsed: {formatTime(elapsed)}</p>
    </div>
  );
}
