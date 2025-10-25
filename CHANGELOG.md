# Changelog

All notable changes to the Acoustic Drone Detection System will be documented in this file.

## [Unreleased] - 2025-10-26

### Changed - Visual Polish for Presentation

**Visualization Generator (`tools/make_visuals.py`):**
- Enabled `constrained_layout` globally to eliminate all text/colorbar overlaps
- Upgraded canvas to 1080p dimensions (19.2×10.8 inches) for presentation displays
- Replaced external colorbars with inset colorbars (3% width) that never overlap plots
- Standardized colormaps across all figures (viridis=mel, coolwarm=mfcc, plasma=spectral)
- Moved parameter distribution panel in architecture diagram to dedicated right-side column
- Added text wrapping (72 chars) for long layer descriptions in architecture diagram
- Implemented multi-format export: `*_1080p.jpg` (slides), `*_A4.pdf` (print), `*.png` (web)

**README Documentation:**
- Fixed parameter count duplication (consolidated to single authoritative value: 2,080,323)
- Added "Performance at a Glance" table with actual validation metrics (97.22% accuracy)
- Updated all metrics to match `evaluation_summary.txt` results
- Added figure captions with numbers (Fig. 1-4) and metadata sidecar links
- Replaced generic performance estimates with measured results from validation set
- Updated image links to point to `*_1080p.jpg` versions for optimal GitHub rendering

**Quality Improvements:**
- Zero clipped titles, labels, or colorbars (verified by visual inspection)
- Figures readable from 2 meters on 16:9 displays
- Consistent typography and spacing across all visualizations
- Professional presentation-ready output with A4 PDF exports for print media

### Technical Details
- Preprocessing flowchart: 5-row grid with optimized height ratios [0.9, 1.0, 2.0, 1.8, 1.6]
- Architecture diagram: Increased vertical spacing, auto-wrapped text, right-panel stats
- Training/system flowcharts: Inset boxes for auxiliary information (inference times)
- All visualizations saved in 3 formats: 150 DPI JPEG (slides), 300 DPI PDF (print), 300 DPI PNG (web)

### Files Modified
- `tools/make_visuals.py`: Complete refactor for presentation quality
- `README.md`: Consolidated metrics, added figure captions, linked metadata
- `CHANGELOG.md`: Created to document visual polish updates

### Generated Assets
New visualization files in `visualizations/`:
- `01_preprocessing_flowchart_1080p.jpg` + `_A4.pdf` + `.png`
- `02_crnn_architecture_1080p.jpg` + `_A4.pdf` + `.png`
- `03_training_pipeline_1080p.jpg` + `_A4.pdf` + `.png`
- `04_complete_system_flowchart_1080p.jpg` + `_A4.pdf` + `.png`
- `class_comparison.png` (300 DPI)
- Corresponding `.meta.json` sidecars for validation

---

## [1.0.0] - 2025-10-25

### Added
- Initial CRNN architecture with Temporal-Frequency Attention
- Multi-channel preprocessing pipeline (Mel + MFCC + Spectral)
- Training pipeline with AdamW optimizer and cosine annealing
- Model evaluation framework with comprehensive metrics
- Dataset validation and audio sample verification
- Visualization generator with model introspection

### Performance
- Validation Accuracy: 97.22%
- Macro F1-Score: 0.9723
- Model Parameters: 2,080,323 (7.9 MB FP32)
- Inference Time: ~65ms (GPU), ~85-100ms (CPU)

### Dataset
- EDTH Munich Acoustic Drone Detection Dataset
- Perfectly balanced: 180 train + 60 val samples per class
- Classes: drone, helicopter, background
- Audio: 44.1kHz → 22.05kHz, 5s → 3s windows

---

*Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)*
