# PanBrida BRD4 novelty audit (existing generation batch)

Source: Google Drive `BRD4_candidate_audit.csv` (1,632 candidate rows) and reference `BRD4_modeling_cohort.csv` (6,027 compounds). This is a pre-existing generated batch, **not** the latest 1,000-molecule GitHub Actions smoke test.

- Candidate rows: 1632
- Novel vs full cohort: 1053 (64.52%)
- Non-novel vs full cohort: 579
- Novel vs model training subset: 1068
- Inside model applicability domain: 1157
- Property filter pass: 1165
- Marked duplicate within generation: 314

Novelty is exact canonical structure non-overlap as recorded in the audit. It does not prove biological activity or global chemical novelty. Source files: https://drive.google.com/file/d/1Ppa7thcLw7pSxSrTzVbxTw9cXs2e0LSS/view and https://drive.google.com/file/d/1YXiAlH2QfBRjiZoS8NqofIE2ZtTbg-wn/view.
