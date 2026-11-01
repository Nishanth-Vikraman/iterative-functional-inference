# IFI v3.1 — Frozen-checkpoint OOD ladder

v3.1 is evaluation-only. It loads the completed v3.0 Phase-A and Phase-B checkpoints,
sets every parameter to `requires_grad=False`, and performs forward functional composition.
It never creates an optimizer.

The canonical target registry is generated deterministically from the persisted v3.0 resolved
configuration. It includes the original held-out coefficient combinations, coefficient
extrapolation at fixed normalized distances, four absent structural primitives, and mild/hard
novel compositions. The registry is written before evaluation and cannot be changed in place.

The exact Bayesian diagnostic oracle operates over the locked target hypotheses within each
bucket/severity. It is used only for an oracle trajectory and to project desired updates into the
frozen `O_phi`. Its posterior, target metadata, and projected actions never enter IFI inputs.

The frozen checkpoints were built with explicit-history positional capacity 12. Evaluation runs
through F18 (three times the Phase-B training horizon of six). At iterations beyond checkpoint
capacity, the controller receives the most recent 12 explicit history entries. No weights or
positional parameters are extended or changed.

Artifacts are written under `artifacts/v3_1`, including exact training support, checkpoint hashes,
the integrity audit, target registry, streaming raw records, projection diagnostics, summaries,
locked classifications, plots, and `REPORT.md`.
