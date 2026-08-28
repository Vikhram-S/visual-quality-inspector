import React from 'react';
import '@testing-library/jest-dom';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from './App';

class DummyFileReader {
  readAsDataURL() {
    this.result = 'data:image/jpeg;base64,dummy';
    if (this.onloadend) this.onloadend();
  }
}

describe('App Frontend Component Tests', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    global.FileReader = DummyFileReader;
  });

  it('renders application header and title', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'ok', model_loaded: true, timestamp: '2026-08-28T00:00:00' }),
    });

    await act(async () => {
      render(<App />);
    });
    expect(screen.getByText(/Image Quality & Defect Detector/i)).toBeInTheDocument();
    expect(screen.getByText(/Audit Single Image/i)).toBeInTheDocument();
  });

  it('handles upload flow rendering loading state then result state', async () => {
    global.fetch = vi.fn().mockImplementation(async (url) => {
      if (url.includes('/api/health')) {
        return {
          ok: true,
          json: async () => ({ status: 'ok', model_loaded: true }),
        };
      }
      if (url.includes('/api/analyze')) {
        return {
          ok: true,
          json: async () => ({
            id: 'test-uuid-1234',
            filename: 'sample.jpg',
            quality_score: 88.5,
            quality_label: 'ACCEPTABLE',
            issues: [],
            image_stats: {
              laplacian_var: 145.2,
              mean_luminance: 120.5,
              shadow_clip_pct: 0.0,
              highlight_clip_pct: 1.2,
              noise_variance: 2.1,
              blockiness_index: 1.05
            },
            explanation: 'Image quality is pristine.',
            heatmap_base64: 'data:image/png;base64,fake',
            created_at: '2026-08-28T00:00:00'
          }),
        };
      }
      return { ok: false };
    });

    let container;
    await act(async () => {
      const res = render(<App />);
      container = res.container;
    });

    const file = new File(['fake-image-data'], 'sample.jpg', { type: 'image/jpeg' });
    const fileInput = container.querySelector('input[type="file"]');
    
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    const auditBtn = screen.getByRole('button', { name: /Run Defect & Quality Audit/i });
    await act(async () => {
      fireEvent.click(auditBtn);
    });

    expect(await screen.findByText('ACCEPTABLE')).toBeInTheDocument();
    expect(await screen.findByText('88.5')).toBeInTheDocument();
  });

  it('renders error state on failed upload request', async () => {
    global.fetch = vi.fn().mockImplementation(async (url) => {
      if (url.includes('/api/health')) {
        return {
          ok: true,
          json: async () => ({ status: 'ok', model_loaded: true }),
        };
      }
      if (url.includes('/api/analyze')) {
        return {
          ok: false,
          status: 400,
          json: async () => ({ detail: 'Analysis engine failure' }),
        };
      }
      return { ok: false };
    });

    let container;
    await act(async () => {
      const res = render(<App />);
      container = res.container;
    });

    const file = new File(['fake-image-data'], 'sample.jpg', { type: 'image/jpeg' });
    const fileInput = container.querySelector('input[type="file"]');
    
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    const auditBtn = screen.getByRole('button', { name: /Run Defect & Quality Audit/i });
    await act(async () => {
      fireEvent.click(auditBtn);
    });

    expect(await screen.findByText(/Analysis engine failure/i)).toBeInTheDocument();
  });
});
