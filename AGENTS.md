# AGENTS.md

## Project intent
This repository is a prototype for endoscopy image quality scoring and enhancement.

## Critical context
- Current heuristics are exploratory, not final.
- Do not treat the current manual rule-based pipeline as ground truth.
- Final goal is scoring + enhancement of endoscopic images.

## Required direction
- A major part of the final solution should come from investigating and integrating existing open-source code and lightweight neural-network-based methods.
- Prefer realistic reusable open-source components over reinventing everything from scratch.
- Do not fall back to naive full-image average IQA as the main decision logic.

## Working rules
- First summarize the current pipeline and propose a plan before major rewrites.
- Keep code modular and explainable.
- Preserve useful prototype components when appropriate.