// src/components/Disclaimer.jsx
export default function Disclaimer() {
  return (
    <aside
      className="disclaimer-banner"
      role="note"
      aria-label="AI disclaimer"
    >
      <span className="disc-icon" aria-hidden="true">⚕️</span>
      <div>
        <strong>AI-Generated Content — Not Medical Advice</strong>
        <p>
          This is an AI-generated simplification and is <strong>not</strong> a substitute for
          professional medical advice, diagnosis, or treatment.
          Always confirm information with your doctor or qualified healthcare provider.
        </p>
      </div>
    </aside>
  );
}
