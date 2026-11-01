# IFI v4.1: Observable Margin Smoke

v4.1 preserves the v4.0 worlds, population replay, controller architecture, capacity,
history memory, reset rules, baselines, and frozen F64 evaluation. It changes only the
preregistered observability and optimization mechanics motivated by the v4.0 failure:

- eight mutually orthogonal real bootstrap probes replace two random probes;
- `softplus(q - 0.95)` replaces squared contraction ratio;
- the action-energy penalty is zero;
- Adam learning rate is `1e-4` and gradient clipping is disabled;
- non-finite losses or gradients abort the run.

The eight action/transition pairs are full rank in the eight-dimensional action space, so the
hidden linear correction matrix is identifiable from bootstrap evidence in principle. They
remain ordinary interactions and expose neither the matrix nor an oracle action.

The locked gate at step 2,000 requires median frozen F1–F8 q below 0.95 and median F8/F0
below 0.75. Failure stops the smoke without tuning. Passing continues the same single seed to
12,000 asynchronous one-step updates and the unchanged F64/history-intervention evaluation.
