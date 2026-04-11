# Research & Evaluation Matrix (Phase 1)

This matrix tracks reusable open-source neural/public codebases prioritized for this project.

| Candidate | Purpose | License* | Integration difficulty | Relevance to this project | Current status |
|---|---|---|---|---|---|
| [IQA-PyTorch / PyIQA](https://github.com/chaofengc/IQA-PyTorch) | Unified IQA framework (NR/FR metrics, training/inference) | Check upstream repo | Medium | Fastest route to integrate neural IQA baselines behind a common API. | **Integrated (Phase 1 baseline path)** |
| [HyperIQA](https://github.com/SSL92/hyperIQA) | Blind IQA baseline model | Check upstream repo | Medium | Good first neural IQA baseline for no-reference quality scoring. | **Enabled via PyIQA metric selection** |
| [EndoViT](https://github.com/DominikBatic/EndoViT) | Endoscopy-domain pretrained visual backbone | Apache-2.0 (as listed upstream) | Medium-High | Reduces domain gap; useful future backbone for patch-level quality/artifact tasks. | Investigating for Phase 2 |
| [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) | Lightweight low-light enhancement | Non-commercial research use (upstream notice) | Medium | Strong candidate for low-light enhancement branch in endoscopic scenes. | **Integrated (Phase 1 optional enhancer)** |
| [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) | Transformer low-light enhancement | MIT (upstream) | High | Higher restoration quality candidate for future compare/ablation. | Investigating for Phase 2 |
| [EAD2020 winning method](https://github.com/ubamba98/EAD2020) | Endoscopic artifact detection/segmentation | MIT | Medium | Directly targets artifact-aware routing and explainable reject decisions. | Interface-ready, not integrated |
| [Hyper-Kvasir resources](https://github.com/simula/hyper-kvasir) | Public GI endoscopy data support | Check upstream repo | Low-Medium | Dataset backbone for evaluation, transfer, and domain validation. | Data/eval planning |

\*Always verify license terms at integration time; upstream licenses can change.

## Phase 1 decisions implemented

- Integrated optional neural IQA baseline path via `pyiqa` (default metric: `hyperiqa`).
- Integrated optional lightweight enhancement baseline path via Zero-DCE wrapper (checkpoint-based).
- Preserved heuristic baseline as the default runnable path.

## Deferred for later phases

- Full artifact model integration (real segmentation/detection network)
- Endoscopy-domain fine-tuning and calibrated decision model
- Multi-model enhancement benchmark and safety constraints
