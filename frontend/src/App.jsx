import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, Image as ImageIcon, CheckCircle, AlertTriangle, XCircle, 
  Activity, History, FileText, Zap, Info, RefreshCw, Layers
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined && import.meta.env.VITE_API_BASE_URL !== '' 
  ? import.meta.env.VITE_API_BASE_URL 
  : (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '');

export default function App() {
  const [activeTab, setActiveTab] = useState('audit'); // 'audit' | 'history' | 'docs'
  const [healthStatus, setHealthStatus] = useState(null);
  
  // Single image audit state
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [auditResult, setAuditResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);

  // History state
  const [historyList, setHistoryList] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotal, setHistoryTotal] = useState(0);

  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealthStatus(data);
      } else {
        setHealthStatus({ status: 'error', model_loaded: false });
      }
    } catch (err) {
      setHealthStatus({ status: 'offline', model_loaded: false });
    }
  };

  const handleFileChange = (file) => {
    setErrorMsg(null);
    setShowHeatmap(false);
    if (!file) return;

    if (file.size > 15 * 1024 * 1024) {
      setErrorMsg("File size exceeds 15MB limit. Please upload a smaller image.");
      return;
    }

    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
      setErrorMsg("Invalid file format. Please upload JPG, PNG, WEBP, BMP, or TIFF images.");
      return;
    }

    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setErrorMsg(null);
    setShowHeatmap(false);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned error ${res.status}`);
      }

      const data = await res.json();
      setAuditResult(data);
    } catch (err) {
      setErrorMsg(err.message || "Failed to analyze image. Ensure backend is running.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fetchHistory = async (page = 1) => {
    setIsLoadingHistory(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/analyses?page=${page}&limit=12`);
      if (res.ok) {
        const data = await res.json();
        setHistoryList(data.items);
        setHistoryTotal(data.total);
        setHistoryPage(data.page);
      }
    } catch (err) {
      console.error("Failed to load history:", err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'history') {
      fetchHistory(1);
    }
  }, [activeTab]);

  return (
    <div className="app-container">
      {/* App Header */}
      <header className="app-header">
        <div className="logo-area">
          <div className="logo-icon">
            <Zap size={24} />
          </div>
          <div>
            <h1 className="logo-title">Image Quality & Defect Detector</h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Explainable AI Visual Quality Audit System
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button 
            className={`tab-btn ${activeTab === 'audit' ? 'active' : ''}`}
            onClick={() => setActiveTab('audit')}
          >
            <Activity size={16} /> Audit Single Image
          </button>
          <button 
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History size={16} /> Analysis History
          </button>
          <button 
            className={`tab-btn ${activeTab === 'docs' ? 'active' : ''}`}
            onClick={() => setActiveTab('docs')}
          >
            <Layers size={16} /> System Specs
          </button>
        </nav>

        {/* Backend Health Indicator */}
        <div className="status-badge">
          <span 
            className="dot" 
            style={{ 
              background: healthStatus?.model_loaded ? 'var(--accent-green)' : 'var(--accent-red)',
              boxShadow: healthStatus?.model_loaded ? '0 0 8px var(--accent-green)' : '0 0 8px var(--accent-red)'
            }}
          />
          {healthStatus?.model_loaded ? 'Engine Online' : 'Connecting...'}
        </div>
      </header>

      {/* Main Workspace Content */}
      {activeTab === 'audit' && (
        <div className="audit-grid">
          {/* Left Column: Image Upload & Controls */}
          <div className="glass-card">
            <h2 className="card-title">
              <Upload size={18} color="var(--accent-blue)" /> Upload Image Target
            </h2>

            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
              onChange={(e) => e.target.files[0] && handleFileChange(e.target.files[0])}
            />

            {!imagePreview ? (
              <div 
                className="dropzone"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <ImageIcon className="drop-icon" />
                <p className="drop-title">Drag & drop image here or click to browse</p>
                <p className="drop-sub">Supports JPG, PNG, WEBP, BMP (Max 15MB)</p>
              </div>
            ) : (
              <div className="preview-container">
                <img 
                  src={showHeatmap && auditResult?.heatmap_base64 ? auditResult.heatmap_base64 : imagePreview} 
                  alt="Preview" 
                  className="img-preview" 
                />

                {auditResult?.heatmap_base64 && (
                  <button 
                    className="btn-secondary" 
                    style={{ 
                      margin: '0.5rem 0', 
                      width: '100%',
                      background: showHeatmap ? 'var(--accent-purple)' : 'rgba(255, 255, 255, 0.08)',
                      borderColor: showHeatmap ? 'var(--accent-purple)' : 'rgba(255, 255, 255, 0.15)'
                    }}
                    onClick={() => setShowHeatmap(!showHeatmap)}
                  >
                    <Layers size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                    {showHeatmap ? 'Hide Defect Heatmap (Show Original)' : 'Show Problem Regions Heatmap'}
                  </button>
                )}

                <div style={{ display: 'flex', gap: '0.5rem', width: '100%' }}>
                  <button 
                    className="btn-secondary" 
                    style={{ flex: 1 }}
                    onClick={() => { setSelectedFile(null); setImagePreview(null); setAuditResult(null); setShowHeatmap(false); }}
                  >
                    Change Image
                  </button>
                </div>
              </div>
            )}

            {errorMsg && (
              <div className="error-banner">
                <AlertTriangle size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                {errorMsg}
              </div>
            )}

            <button 
              className="btn-primary"
              disabled={!selectedFile || isAnalyzing}
              onClick={runAnalysis}
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="spinner" size={18} style={{ border: 'none', width: '18px', height: '18px' }} />
                  Extracting Features & Analyzing...
                </>
              ) : (
                <>
                  <Zap size={18} /> Run Defect & Quality Audit
                </>
              )}
            </button>
          </div>

          {/* Right Column: Analysis Results */}
          <div className="glass-card">
            <h2 className="card-title">
              <Activity size={18} color="var(--accent-cyan)" /> Quality Audit Diagnostics
            </h2>

            {!auditResult ? (
              <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
                <Info size={40} style={{ opacity: 0.3, marginBottom: '0.75rem' }} />
                <p>Upload an image and click "Run Defect & Quality Audit" to view scores, issue detection, and feature breakdowns.</p>
              </div>
            ) : (
              <div>
                {/* Score & Label Header */}
                <div className="score-card">
                  <div className="gauge-wrapper">
                    <svg className="gauge-svg" width="100" height="100" viewBox="0 0 100 100">
                      <circle className="gauge-bg" cx="50" cy="50" r="45" />
                      <circle 
                        className="gauge-val" 
                        cx="50" 
                        cy="50" 
                        r="45"
                        style={{
                          stroke: auditResult.quality_score >= 75 ? 'var(--accent-green)' : (auditResult.quality_score >= 45 ? 'var(--accent-amber)' : 'var(--accent-red)'),
                          strokeDashoffset: 283 - (283 * auditResult.quality_score) / 100
                        }}
                      />
                    </svg>
                    <span className="gauge-text">{auditResult.quality_score}</span>
                  </div>

                  <div className="label-wrapper">
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Overall Status</span>
                    <span className={`quality-badge ${auditResult.quality_label}`}>
                      {auditResult.quality_label === 'ACCEPTABLE' && <CheckCircle size={16} />}
                      {auditResult.quality_label === 'DEGRADED' && <AlertTriangle size={16} />}
                      {auditResult.quality_label === 'DEFECTIVE' && <XCircle size={16} />}
                      {auditResult.quality_label}
                    </span>
                  </div>
                </div>

                {/* Detected Issues Section */}
                <h3 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                  Detected Issues ({auditResult.issues.length})
                </h3>
                {auditResult.issues.length === 0 ? (
                  <p style={{ fontSize: '0.85rem', color: 'var(--accent-green)', marginBottom: '1.25rem' }}>
                    <CheckCircle size={14} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                    No major defects detected. Image meets clean quality standards.
                  </p>
                ) : (
                  <div className="issues-list">
                    {auditResult.issues.map((iss, idx) => (
                      <div key={idx} className="issue-item">
                        <div className="issue-left">
                          <span className="issue-type">{iss.type}</span>
                          <span className={`severity-tag ${iss.severity}`}>{iss.severity}</span>
                        </div>
                        <span className="confidence-text">
                          Conf: {(iss.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Image Features & Statistics Grid */}
                <h3 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                  Engineered Feature Metrics
                </h3>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Sharpness (Laplacian)</div>
                    <div className="stat-value">{auditResult.image_stats.laplacian_var}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Luminance (Mean)</div>
                    <div className="stat-value">{auditResult.image_stats.mean_luminance}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Shadow Clip %</div>
                    <div className="stat-value">{auditResult.image_stats.shadow_clip_pct}%</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Highlight Clip %</div>
                    <div className="stat-value">{auditResult.image_stats.highlight_clip_pct}%</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Noise Variance</div>
                    <div className="stat-value">{auditResult.image_stats.noise_variance}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Blockiness Index</div>
                    <div className="stat-value">{auditResult.image_stats.blockiness_index}</div>
                  </div>
                </div>

                {/* Explainability Panel */}
                <h3 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                  Decision Rationale & Explainability
                </h3>
                <div className="explanation-box">
                  <Info size={16} style={{ float: 'left', marginRight: '8px', color: 'var(--accent-purple)' }} />
                  {auditResult.explanation}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* History Tab Content */}
      {activeTab === 'history' && (
        <div className="glass-card">
          <h2 className="card-title">
            <History size={18} color="var(--accent-blue)" /> Stored Analysis Records ({historyTotal})
          </h2>

          {isLoadingHistory ? (
            <div className="loading-shimmer">
              <div className="spinner" />
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Loading analysis records...</p>
            </div>
          ) : historyList.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
              No previous analyses stored. Upload images in the Audit tab to record results.
            </p>
          ) : (
            <div className="history-grid">
              {historyList.map((item) => (
                <div key={item.id} className="history-card" onClick={() => { setAuditResult(item); setActiveTab('audit'); }}>
                  <img 
                    src={`${API_BASE_URL}/api/images/${item.id}`} 
                    alt={item.filename} 
                    className="history-img"
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '170px' }}>
                      {item.filename}
                    </span>
                    <span className={`quality-badge ${item.quality_label}`} style={{ padding: '0.15rem 0.5rem', fontSize: '0.7rem' }}>
                      {item.quality_score}
                    </span>
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {new Date(item.created_at).toLocaleString()}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* System Specs & Docs Tab */}
      {activeTab === 'docs' && (
        <div className="glass-card">
          <h2 className="card-title">
            <FileText size={18} color="var(--accent-purple)" /> System Architecture & Technical Specifications
          </h2>
          <div style={{ lineHeight: '1.7', color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            <h3 style={{ color: 'var(--text-primary)', marginTop: '1rem', marginBottom: '0.5rem' }}>Hardware-Optimized Architecture</h3>
            <p>
              This system is built specifically to operate under strict hardware constraints (8GB RAM, CPU-only). 
              Instead of relying on heavy deep neural networks, it utilizes a hybrid classical CV feature extraction pipeline paired 
              with a scikit-learn Random Forest ensemble.
            </p>

            <h3 style={{ color: 'var(--text-primary)', marginTop: '1.25rem', marginBottom: '0.5rem' }}>Extracted Feature Vector</h3>
            <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
              <li><strong>Sharpness:</strong> Variance of Laplacian, Tenengrad Sobel magnitude, FFT high-frequency ratio</li>
              <li><strong>Exposure:</strong> Mean luminance, Shadow clip %, Highlight clip %</li>
              <li><strong>Noise:</strong> High-frequency residual variance, Immerkaer fast noise estimation</li>
              <li><strong>Blockiness & Integrity:</strong> 8x8 DCT boundary edge index, Shannon entropy</li>
            </ul>

            <h3 style={{ color: 'var(--text-primary)', marginTop: '1.25rem', marginBottom: '0.5rem' }}>Evaluation Summary</h3>
            <p>
              Evaluated on both unseen synthetic test splits and real-world holdout photographs (Unsplash/COCO).
              Achieves 92.38% synthetic test accuracy and 84.79% real-world holdout accuracy.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
