# Endoscopy IQA Assistant

## What this project is

This is an exploratory prototype for endoscopy image quality assessment (IQA) and enhancement.

The current implementation started with manual heuristics and now enters a **controlled Phase 1** with optional neural baselines.

⚠️ Important:
- The heuristic pipeline is a baseline and **NOT** ground truth.
- A major part of final system capability must come from existing open-source neural methods.

---

## Final Goal

1. Score endoscopic images
2. Detect locally degraded but clinically important regions
3. Decide: accept / enhance / reject
4. Enhance low-quality images
5. Integrate reusable open-source methods and lightweight neural networks

---

## Current Phase 1 Scope (controlled)

- Keep heuristic baseline runnable
- Fix obvious broken parts without large rewrite
- Add clean modular interfaces for IQA / artifact / enhancer
- Integrate exactly one neural IQA baseline (optional PyIQA/HyperIQA path)
- Integrate exactly one lightweight enhancer baseline (optional Zero-DCE path)
- Add research/evaluation matrix for candidate open-source codebases

---

## Repository Structure (Phase 1)

- `main.py`
  - CLI runner for heuristic vs neural IQA mode
  - optional enhancer path
  - saves visualization outputs to files (non-GUI flow)

- `models/`
  - `interfaces.py`: `IQAModel`, `ArtifactModel`, `Enhancer` interfaces
  - `heuristic_iqa.py`: current heuristic baseline wrapper
  - `pyiqa_model.py`: optional neural IQA baseline via `pyiqa`

- `artifacts/`
  - `null_artifact_model.py`: Phase 1 placeholder with unified interface

- `enhancers/`
  - `zero_dce_enhancer.py`: optional Zero-DCE enhancer wrapper (checkpoint-based)

- `scorer.py`, `patch_scorer.py`
  - existing heuristic components retained

- `batch_*.py`, `Batch_patch_score.py`
  - baseline calibration scripts retained
  - plus `batch_patch_score.py` lowercase entrypoint for compatibility/convention

- `docs/research_matrix.md`
  - open-source candidate matrix (IQA, enhancement, artifact)

---

## Install

```bash
pip install -r requirements.txt
```

If you want optional neural baselines (PyIQA/Zero-DCE path), install:

---

## How to run

### 1) Heuristic baseline (default)

```bash
python main.py --image data/calibration/good/202511241219G061.jpg --iqa-mode heuristic --enhancer none --output-dir outputs/heuristic
```

### 2) Neural IQA baseline (PyIQA/HyperIQA)

```bash
python main.py --image data/calibration/good/202511241219G061.jpg --iqa-mode pyiqa --pyiqa-metric hyperiqa --enhancer none --output-dir outputs/pyiqa
```

### 2b) Fusion IQA (heuristic + neural)

```bash
python main.py --image data/calibration/good/202511241219G061.jpg --iqa-mode fusion --pyiqa-metric hyperiqa --fusion-w1 0.7 --fusion-w2 0.3 --enhancer none --output-dir outputs/fusion
```

Fusion currently uses:
`fusion_score = w1 * heuristic + w2 * neural`
and writes `fusion_breakdown` (score, components, weights) to `report.json`.

### 3) Optional Zero-DCE enhancement + heuristic IQA

```bash
python main.py --image data/calibration/good/202511241219G061.jpg --iqa-mode heuristic --enhancer zero_dce --zero-dce-checkpoint /path/to/zero_dce.pth --output-dir outputs/zero_dce
```

If no checkpoint is provided for Zero-DCE, enhancer runs in safe fallback mode (input image unchanged) and reports warning metadata.

### 4) Compare heuristic vs neural vs fusion IQA (same image)

```bash
python compare_iqa_modes.py --image data/calibration/good/202511241219G061.jpg --output outputs/compare_iqa.json --pyiqa-metric hyperiqa --fusion-w1 0.7 --fusion-w2 0.3
```

This writes all three runs into one JSON for side-by-side inspection.

### 5) Small-subset smoke test (Phase 1 only)

```bash
python scripts/phase1_smoke_test.py --max-images 3 --output outputs/phase1_smoke/smoke_summary.json
```

This uses only a tiny subset from `data/calibration` and writes summary JSON to file.

---

## Visualization behavior in cloud / headless environments

All visual outputs are saved to files under `--output-dir`:

- `worst_patches.png`
- `original_and_fov.png`
- `enhanced_image.png` (if enhancer enabled)
- `report.json`

No interactive display window is required.

---

## Current limitations (still unresolved)

- Artifact model is interface-only placeholder in Phase 1.
- Neural IQA and Zero-DCE are optional and dependency/checkpoint sensitive.
- Heuristic components remain exploratory.
- Decision engine (accept/enhance/reject) is not yet trained/calibrated.

---

## Next direction (after Phase 1 approval)

- Add real artifact segmentation model integration
- Calibrate and compare IQA model families on validation data
- Add robust decision logic with confidence and explainability
- Establish enhancement safety constraints and clinician review loop
