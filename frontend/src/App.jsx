// src/App.jsx
import './App.css';
import { useState, useEffect } from 'react';
import ProgressIndicator from './components/ProgressIndicator.jsx';
import ResultsView from './components/ResultsView.jsx';
import { uploadDocument, startSimplify, getJob, getHealth } from './api.js';

export default function App() {
  const [health, setHealth] = useState(null);
  
  // App state: 'upload' | 'extracted' | 'processing' | 'results'
  const [view, setView] = useState('upload');
  
  // Data state
  const [textInput, setTextInput] = useState('');
  const [docId, setDocId] = useState(null);
  const [readingLevel, setReadingLevel] = useState('basic');
  const [jobId, setJobId] = useState(null);
  const [jobStage, setJobStage] = useState('queued');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // Drag & drop state
  const [isDragOver, setIsDragOver] = useState(false);
  
  // Upload status
  const [isUploading, setIsUploading] = useState(false);

  // Initial health check
  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'offline' }));
  }, []);

  // Polling logic when processing
  useEffect(() => {
    let timer;
    async function poll() {
      if (view !== 'processing' || !jobId) return;
      try {
        const data = await getJob(jobId);
        setJobStage(data.stage || 'queued');
        
        if (data.status === 'done') {
          setResult(data.result);
          setView('results');
        } else if (data.status === 'failed') {
          setError(data.error || 'The job failed during processing.');
          setView('upload');
        } else {
          // Keep polling
          timer = setTimeout(poll, 2000);
        }
      } catch (err) {
        setError(err.message || 'Lost connection to server while polling.');
        setView('upload');
      }
    }
    
    if (view === 'processing') {
      poll();
    }
    return () => clearTimeout(timer);
  }, [view, jobId]);

  // Handlers
  const handleFileUpload = async (file) => {
    if (!file) return;
    setIsUploading(true);
    setError(null);
    try {
      const data = await uploadDocument(file);
      setDocId(data.doc_id);
      setTextInput(data.extracted_text);
      setView('extracted');
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };
  
  const handleDrop = (e) => {
    e.preventDefault(); setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  };
  const handleFileInput = (e) => handleFileUpload(e.target.files[0]);

  const handleSimplify = async () => {
    if (!textInput.trim()) {
      setError("Please provide some text to simplify.");
      return;
    }
    setError(null);
    try {
      // Send the currently edited text (ignoring docId if the text was modified heavily, 
      // but the API prefers text if provided directly)
      const data = await startSimplify({ text: textInput, readingLevel });
      setJobId(data.job_id);
      setJobStage('queued');
      setView('processing');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleReset = () => {
    setTextInput('');
    setDocId(null);
    setJobId(null);
    setResult(null);
    setError(null);
    setView('upload');
  };

  // Header status indicator
  const isOk = health?.status === 'ok';
  const isOffline = health?.status === 'offline';

  return (
    <div className="app-wrapper">
      {/* ── Header ── */}
      <header className="app-header">
        <a href="/" className="app-logo">
          <div className="logo-icon">M</div> MedSimplify
        </a>
        <div className="header-status">
          <div className={`status-dot ${isOffline ? 'offline' : !health ? 'checking' : ''}`} />
          <span>
            {isOffline ? 'Backend offline' : !health ? 'Checking connection...' : 
             `Server ${health.status} ${health.mock_mode ? '(Mock Mode)' : ''}`}
          </span>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="app-content">


        {/* Error Banner */}
        {error && (
          <div className="error-banner slide-in" role="alert">
            <span className="error-icon" aria-hidden="true">⚠️</span>
            <div>
              <strong>Error occurred</strong>
              <p>{error}</p>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => setError(null)} style={{ marginLeft: 'auto' }}>Dismiss</button>
          </div>
        )}

        {/* ── VIEW: Upload ── */}
        {view === 'upload' && (
          <div className="upload-screen fade-in">
            <h1>Simplify Medical Documents</h1>
            <p className="upload-subtitle">Translate complex clinical text into plain language instantly.</p>
            
            <div 
              className={`dropzone ${isDragOver ? 'dragover' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              aria-label="Upload document area"
            >
              {isUploading ? (
                <>
                  <div className="spinner" style={{ marginBottom: 16 }} />
                  <h3>Extracting text...</h3>
                </>
              ) : (
                <>
                  <span className="dropzone-icon" aria-hidden="true">📄</span>
                  <h3>Drag & drop a medical document</h3>
                  <p>Or click to browse files</p>
                  <div className="file-types">
                    <span className="badge badge-teal">PDF</span>
                    <span className="badge badge-teal">DOCX</span>
                    <span className="badge badge-blue">TXT</span>
                    <span className="badge badge-warning">PNG/JPG</span>
                  </div>
                  <input type="file" onChange={handleFileInput} accept=".pdf,.docx,.txt,.png,.jpg,.jpeg" aria-hidden="true" title="Upload document" />
                </>
              )}
            </div>
            
            <div className="divider">OR</div>
            
            <div className="paste-area">
              <label htmlFor="manual-text">Paste clinical text directly</label>
              <textarea 
                id="manual-text"
                placeholder="Patient presents with hypertension and hyperlipidemia. Prescribed Atorvastatin 20mg..."
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
              />
              <div className="paste-submit">
                <button 
                  className="btn btn-secondary" 
                  disabled={!textInput.trim() || isUploading} 
                  onClick={() => setView('extracted')}
                >
                  Continue with text →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW: Extracted / Ready to simplify ── */}
        {view === 'extracted' && (
          <div className="extracted-view fade-in">
            {docId && (
              <div className="extracted-banner">
                <span aria-hidden="true">✅</span>
                <span>Text successfully extracted from the document. Please review or edit it before continuing.</span>
              </div>
            )}
            
            <div className="card" style={{ marginBottom: 24 }}>
              <div className="card-header" style={{ justifyContent: 'space-between' }}>
                <h3 style={{ fontSize: '0.95rem' }}>Review Original Text</h3>
                <span className="badge badge-purple">{textInput.split(/\s+/).filter(Boolean).length} words</span>
              </div>
              <div className="card-body">
                <textarea 
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  aria-label="Review extracted text"
                />
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 style={{ fontSize: '0.95rem' }}>Simplification Settings</h3>
              </div>
              <div className="card-body extracted-actions">
                <div>
                  <label id="rl-label" style={{ marginBottom: 12 }}>Target Reading Level</label>
                  <div className="reading-level-group" role="group" aria-labelledby="rl-label">
                    {['basic', 'intermediate', 'advanced'].map(lvl => (
                      <button 
                        key={lvl}
                        className={`level-btn ${readingLevel === lvl ? 'active' : ''}`}
                        onClick={() => setReadingLevel(lvl)}
                        aria-pressed={readingLevel === lvl}
                      >
                        {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
                        <div className="level-desc" style={{ color: readingLevel === lvl ? 'var(--accent)' : '' }}>
                          {lvl === 'basic' && '5th-6th grade approx.'}
                          {lvl === 'intermediate' && '8th-9th grade approx.'}
                          {lvl === 'advanced' && '11th-12th grade approx.'}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 10, marginTop: 16 }}>
                  <button className="btn btn-ghost" onClick={() => setView('upload')}>Cancel</button>
                  <button className="btn btn-primary btn-lg" onClick={handleSimplify} disabled={!textInput.trim()}>
                    ✨ Simplify Document
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW: Processing ── */}
        {view === 'processing' && (
          <ProgressIndicator stage={jobStage} readingLevel={readingLevel} />
        )}

        {/* ── VIEW: Results ── */}
        {view === 'results' && result && (
          <ResultsView originalText={result.original_text || textInput} result={result} onReset={handleReset} />
        )}

      </main>
    </div>
  );
}
