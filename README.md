# Endoscopy IQA Assistant

## What this project is

This is an exploratory prototype for endoscopy image quality assessment (IQA) and enhancement.

The current implementation is built manually from scratch and mainly uses heuristic and region-based methods to explore the problem.

⚠️ Important:  
The current pipeline is NOT the final intended solution.  
It is only a stepping stone to understand the problem and failure modes.

---

## Final Goal

The final goal of this project is:

1. Score endoscopic images
2. Detect locally degraded but clinically important regions
3. Decide:
   - accept
   - enhance
   - reject
4. Build an enhancement pipeline for low-quality images
5. Integrate existing open-source methods and lightweight neural networks

---

## Key Ideas (Current Understanding)

- Global IQA is insufficient
- Region importance > global average
- Dark / low-detail lumen regions are critical
- Patch-based analysis is more meaningful than full-image scoring

---

## Current Pipeline

Current modules include:

- `scorer.py`
  - blur score
  - exposure score

- `patch_scorer.py`
  - patch-based quality scoring
  - worst region detection

- `utils.py`
  - FOV mask
  - preprocessing

- `main.py`
  - visualization + debugging

---

## Limitations

- Heuristic-based, not robust
- No learning-based model
- Region detection is imperfect
- Some important dark regions may be missed
- Some normal regions may be misclassified

---

## Next Direction

This project should evolve toward:

- Better region proposal
- More stable patch scoring
- Decision module (accept / enhance / reject)
- Integration of open-source IQA / enhancement methods
- Lightweight neural-network-based approaches

---

## Important Constraint

Do NOT treat the current heuristic logic as ground truth.

A major part of future work must come from:
- existing open-source code
- neural-network-based methods
- real IQA / enhancement pipelines

---

## Status

Prototype stage.