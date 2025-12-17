# Doodle Recognition Presentation

## Overview
This Beamer presentation provides a comprehensive overview of the Gamified Doodle Recognition project for English Vocabulary Acquisition using Deep Learning.

## Files
- `doodle_recognition_presentation.tex` — Main presentation source file

## Compilation

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages (usually included): beamer, tikz, graphicx, amsmath, booktabs, listings

### Compilation Commands
```bash
cd docs/presentation
pdflatex doodle_recognition_presentation.tex
pdflatex doodle_recognition_presentation.tex  # Run twice for references
```

Or using `latexmk`:
```bash
latexmk -pdf doodle_recognition_presentation.tex
```

## Presentation Structure

1. **Introduction** — Background, motivation, objectives
2. **Dataset** — QuickDraw dataset overview, sample categories
3. **Methodology** — ResNet18 architecture, OpenCV similarity, preprocessing
4. **Results** — Model performance, per-class analysis
5. **Web Application** — Features, gamification
6. **Conclusion** — Achievements, future work

## Image Dependencies

The presentation references images from `../proposal-dl/`:
- `airplane_sample.png`
- `cat_sample.png`
- `app_demo.png`

## Notes

- Presentation duration: ~20-25 minutes
- Aspect ratio: 16:9 (widescreen)
- Theme: Madrid with BINUS blue color scheme
