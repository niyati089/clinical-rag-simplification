// src/components/ActionBar.jsx
import { useState } from 'react';
import { jsPDF } from 'jspdf';

function buildPlainText(result) {
  const lines = [];

  if (result.sections?.simple_explanation)
    lines.push('=== Simple Explanation ===\n' + result.sections.simple_explanation);
  if (result.sections?.important_instructions)
    lines.push('\n=== Important Instructions ===\n' + result.sections.important_instructions);
  if (result.sections?.medication_guidance)
    lines.push('\n=== Medication Guidance ===\n' + result.sections.medication_guidance);
  if (result.sections?.follow_up)
    lines.push('\n=== Follow-up Steps ===\n' + result.sections.follow_up);

  if (!lines.length && result.simplified_text)
    lines.push(result.simplified_text);

  // Glossary
  const glossary = result.glossary || {};
  if (Object.keys(glossary).length) {
    lines.push('\n\n=== Glossary ===');
    Object.entries(glossary).forEach(([term, def]) => {
      lines.push(`\n${term}: ${def}`);
    });
  }

  lines.push('\n\n---\nThis is an AI-generated simplification and is not medical advice. Always confirm with your doctor.');
  return lines.join('\n');
}

function buildPDF(result) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const margin = 40;
  const pageWidth = doc.internal.pageSize.getWidth() - margin * 2;
  let y = margin;

  const addLine = (text, opts = {}) => {
    const { size = 11, bold = false, color = [30, 30, 30] } = opts;
    doc.setFontSize(size);
    doc.setFont('helvetica', bold ? 'bold' : 'normal');
    doc.setTextColor(...color);
    const lines = doc.splitTextToSize(text, pageWidth);
    lines.forEach(line => {
      if (y > doc.internal.pageSize.getHeight() - margin) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += size * 1.5;
    });
    y += 6;
  };

  addLine('MedSimplify — Simplified Medical Document', { size: 16, bold: true, color: [0, 150, 130] });
  addLine(new Date().toLocaleDateString(), { size: 9, color: [120, 120, 120] });
  y += 10;

  const sectionDefs = [
    ['Simple Explanation', result.sections?.simple_explanation],
    ['Important Instructions', result.sections?.important_instructions],
    ['Medication Guidance', result.sections?.medication_guidance],
    ['Follow-up Steps', result.sections?.follow_up],
  ];
  sectionDefs.forEach(([title, content]) => {
    if (!content) return;
    addLine(title, { size: 13, bold: true, color: [0, 100, 90] });
    addLine(content, { size: 11 });
    y += 4;
  });

  if (!sectionDefs.some(([, c]) => c) && result.simplified_text) {
    addLine('Simplified Text', { size: 13, bold: true, color: [0, 100, 90] });
    addLine(result.simplified_text, { size: 11 });
  }

  const glossary = result.glossary || {};
  if (Object.keys(glossary).length) {
    y += 8;
    addLine('Glossary', { size: 13, bold: true, color: [0, 100, 90] });
    Object.entries(glossary).forEach(([term, def]) => {
      addLine(`${term}: ${def}`, { size: 10, color: [60, 60, 60] });
    });
  }

  y += 16;
  addLine(
    'This is an AI-generated simplification and is not medical advice. Always confirm with your doctor.',
    { size: 9, color: [150, 90, 0] }
  );

  return doc;
}

export default function ActionBar({ result }) {
  const [copyState, setCopyState] = useState('idle'); // 'idle' | 'success'

  const plainText = buildPlainText(result);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(plainText);
      setCopyState('success');
      setTimeout(() => setCopyState('idle'), 2000);
    } catch (_) {
      // Fallback
      const el = document.createElement('textarea');
      el.value = plainText;
      document.body.appendChild(el);
      el.select();
      document.execCommand('copy');
      document.body.removeChild(el);
      setCopyState('success');
      setTimeout(() => setCopyState('idle'), 2000);
    }
  };

  const handleDownloadTxt = () => {
    const blob = new Blob([plainText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'simplified-medical-doc.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPDF = () => {
    const doc = buildPDF(result);
    doc.save('simplified-medical-doc.pdf');
  };

  return (
    <div className="action-bar" role="toolbar" aria-label="Document actions">
      <span className="action-label">Export simplified document:</span>

      <button
        id="action-copy"
        className={`btn btn-secondary ${copyState === 'success' ? 'btn-copy-success' : ''}`}
        onClick={handleCopy}
        aria-label="Copy simplified text to clipboard"
        title="Copy to clipboard"
      >
        <span aria-hidden="true">{copyState === 'success' ? '✓' : '📋'}</span>
        {copyState === 'success' ? 'Copied!' : 'Copy'}
      </button>

      <button
        id="action-download-pdf"
        className="btn btn-secondary"
        onClick={handleDownloadPDF}
        aria-label="Download as PDF"
        title="Download PDF"
      >
        <span aria-hidden="true">📄</span>
        Download PDF
      </button>

      <button
        id="action-download-txt"
        className="btn btn-secondary"
        onClick={handleDownloadTxt}
        aria-label="Download as plain text"
        title="Download TXT"
      >
        <span aria-hidden="true">📝</span>
        Download TXT
      </button>
    </div>
  );
}
