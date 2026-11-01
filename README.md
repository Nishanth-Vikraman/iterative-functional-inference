# IFI v2 / Functional-Learning v1 — RKHS Teacher

This repository is a clean scientific implementation of Iterative Functional Inference (IFI): a
student learns a reusable **learning action**, then repeatedly applies it to its own current function
without gradients or a teacher at test time. It deliberately does not train an end-to-end query
predictor.

## What is separated

Phase A learns the language of functional modifications:

`O_phi(z) = z + sum_k phi_k B_k(z)`

The learned nonlinear bases `B_k: R^d -> R^d` are shared, `phi=0` is exactly identity, and every
operator preserves dimension. For each analytical teacher update, independently sampled probe inputs
define a function-space projection. Once Phase A is saved, the operator is frozen permanently.

Phase B learns how to choose modifications. A data set encoder, residual/counterfactual encoder, and
small recurrent autoregressive controller predict the frozen coordinates. Each local target is the
projection of one teacher update from the student's **actual** current function. The next state is made
with the student's action and detached before further supervision.

```text
support D ──> E_D ───────────────────────────────┐
                                                 v
(D, F_i(D), residuals) ──> E_R ──> controller ──> phi_i
                              history/phi_<i ────┘   │
                                                    v
             F_{i+1} = O_{phi_i} o F_i <────────────┘
```

There is no `q` representation, separate hypernetwork `H`, query-label training objective, or
test-time optimizer.

## Modules

- `ifi/teachers/base.py`: minimal swappable `Teacher` and `FunctionalUpdate` interfaces.
- `ifi/teachers/rkhs.py`: deterministic one-step RBF RKHS functional-gradient teacher using support
  data only.
- `ifi/operators.py`: dimensionality-preserving operator bases and function-space projection.
- `ifi/phase_a.py`: alternating coordinate inference / shared basis learning and span diagnostics.
- `ifi/models.py`, `ifi/phase_b.py`: residual policy, AR-no-feedback and static matched baselines,
  with on-policy local supervision.
- `ifi/inference.py`: frozen, teacher-free, gradient-free test-time adaptation.
- `ifi/evaluation.py`: novelty splits, K/seeds, RKHS/F0 baselines, action imitation, rollouts, and
  controlled residual interventions.

Adding an SGD, MAML, Bayesian, or another functional-gradient teacher requires implementing only
`Teacher.propose_update`; IFI consumes the returned function evaluations and no teacher internals.

## Run

Python 3.10+ and a CUDA-enabled PyTorch installation are required for the canonical experiment.

```powershell
python -m pip install -e ".[dev]"
ifi all --config configs/canonical.json
```

The canonical configuration explicitly requires CUDA and will fail instead of silently falling back
to CPU. A tiny CPU integration run is available for engineering verification:

```powershell
ifi all --config configs/smoke.json
python -m pytest
```

Use `configs/cuda_smoke.json` for the same quick end-to-end check on the GPU.

Individual stages are restartable:

```powershell
ifi phase-a --config configs/canonical.json
ifi phase-b --config configs/canonical.json
ifi evaluate --config configs/canonical.json
```

## Outputs

Each run writes resolved configuration and checkpoints plus:

- Phase-A reconstruction/projection residual, basis condition number, action statistics, held-out
  reconstruction, and outside-span fraction for every novelty split;
- per-seed raw evaluation CSV and aggregated JSON for action MSE, function error by iteration, K,
  novelty split, and all matched baselines;
- causal intervention data that branches only current residual evidence while holding `F_i`, `z_D`,
  prior actions, and recurrent state fixed;
- mechanistic plots and `REPORT.md` with literal pre-registered checks rather than automatic positive
  claims.

The initial canonical v1 keeps the differential meta-loss off (`lambda_differential=0`). It is fully
implemented and can be enabled in a derived config.

## IFI v2.1 direct residual ablation

The frozen-language follow-up lives in `ifi/v21_*`. It removes the standalone residual encoder and
feeds every six-field support residual token directly to a matched Transformer controller. The three
equal-parameter variants expose current residuals only, residuals plus action history, or residuals
plus action history and persistent Transformer learning state. It reuses `runs/canonical/phase_a.pt`
without modification and compares against the completed historical canonical run.

```powershell
python -m ifi.v21_cli all --config configs/v2_1_direct.json
```

Training writes a rolling atomic checkpoint and flushed log records every 100 steps. Evaluation also
writes cell-level progress, avoiding the silent period of the original canonical evaluator.

## IFI v2.2 predictive recurrent state

`ifi/v22_*` separates action controller `T` from state updater `U`. Teacher action and functional
losses update only `T`; `U` receives only an InfoNCE objective pairing its next state with the actual
next residual from the same trajectory. Negatives are other episodes at the same rollout step. The
contrastive residual signature and state alignment are deterministic and fixed, with no semantic
residual encoder or learned prediction decoder. Gradient routing is measured explicitly.

```powershell
python -m ifi.v22_cli all --config configs/v2_2_predictive.json
```

The canonical CLI uses an exclusive one-shot lock so the preregistered run cannot accidentally be
repeated or tuned after observing its result.

## IFI v2.3 forward-trained recurrent state

`ifi/v23_*` replaces the separate v2.2 state updater with one recurrent Transformer containing a
shared evidence trunk, a current-action head, and a persistent-state head. Training uses locked
100-step coordinate blocks. H-blocks train only the state head through the usefulness of its state
to a frozen next-step action evaluator; Phi-blocks train the shared/action path from the existing
teacher-action and functional-consistency objectives while prior recurrent states remain detached.
The operator and functional transition are stop-gradient boundaries in both cases. There is no
standalone residual encoder, state updater, InfoNCE loss, or residual-prediction target.

```powershell
python -m ifi.v23_cli all --config configs/v2_3_forward.json
```

The primary evaluation remains F0 through F8 and also records a frozen, gradient-free extension to
F16, with separately controlled residual and recurrent-state interventions. The one-shot lock is
created before canonical training begins.

## IFI v3.0 history-necessary Bayesian latent teacher

`ifi/v30_*` is a separate experiment family with a 32-hypothesis exact Bayesian teacher and
sequential evidence batches. The current controller input contains only the newest observable
batch; the generic teacher receives the accumulated observable support and returns its exact
posterior-predictive function through the same `FunctionalUpdate` interface. Latent identity,
posterior probabilities, and teacher features are never exposed to IFI.

Before Phase A or B, the locked gate measures residual/history aliasing along ideal Bayesian
trajectories. Training is forbidden unless every gate criterion passes. Phase A then learns a new
Bayesian-update operator language and freezes it. Phase B compares current evidence alone,
full-history causal self-attention, current-residual cross-attention into history, and AR
no-feedback. There is no semantic residual encoder, recurrent state, state updater, or memory loss.

```powershell
python -m ifi.v30_gate --config configs/v3_0_bayesian.json
python -m ifi.v30_cli all --config configs/v3_0_bayesian.json
```

The canonical report keeps held-out latent rollouts, K/seed results, current-residual controls, and
all explicit-history corruptions separate. v2.x checkpoints and run directories are never reused or
overwritten.

## IFI v4 — population-replay residual learner

v4 is a deliberately small, separate falsification benchmark. It removes the Phase-A operator and
teacher-imitation confounds and asks whether a controller can discover a stationary correction law
from consequences alone. Each hidden world samples a fixed, unobserved, well-conditioned matrix
`B`, and interaction follows

```text
R_{i+1} = R_i + B u_i
u_i     = T(R_i, H_i, u_{i-1})
```

`B` is never supplied to the controller. Eight-dimensional residuals and actions are used, and the
action itself is the functional operator. New worlds begin with real bounded probe interactions so
the position-free query-history controller can infer how actions affect residuals. Its history is a
set of genuine `(R_before, u, R_after)` transitions; there are no positional embeddings, iteration
inputs, world identifiers, teacher actions, oracle targets, or recurrent hidden state.

### Local consequence training

Training uses asynchronous population replay. Every SGD example is a randomly selected real learner
state followed by exactly one world transition. Stored histories and prior transitions are detached,
so gradients flow only through

```text
T -> current action -> current world transition -> one-step loss
```

No ordered multi-step loss or BPTT is used. v4.0 initially used the scale-free contraction ratio

```text
L_contract = ||R_{i+1}||^2 / (||R_i||^2 + epsilon)
```

with a weak action-energy penalty. That combination encouraged a cautious, near-zero-action solution.
The successful v4.1 smoke removed the action penalty, increased the real bootstrap to eight probes,
and used the margin progress objective

```text
q          = ||R_{i+1}|| / (||R_i|| + epsilon)
L_progress = softplus(q - 0.95)
```

Later convergence experiments retained this objective and added only a no-regression consequence
penalty where specified:

```text
L_safe = max(0, ||R_{i+1}||^2 - ||R_i||^2)
L      = L_progress + lambda_safe L_safe
```

Thus v4 learns from whether its proposed action actually contracts the residual—not from an
analytical action, a target function, or an imitation coordinate.

### Experiment sequence and result

- **v4.0:** established the population-replay implementation but converged to small safe actions.
- **v4.1:** learned strong history-conditioned early correction on unseen same-family worlds, yet a
  persistent action floor caused post-minimum divergence during long frozen rollouts.
- **v4.2–v4.4:** tested scale-balanced state coverage, convergence replay, joint-from-scratch
  training, and a capacity ablation. These did not jointly preserve early identification and obtain
  asymptotic stability.
- **v4.5:** aggregated long histories generated by the current policy. It partially improved the
  late action direction but did not remove F64/F128 drift.
- **v4.5-Q:** froze the strongest proposal controller and tested a separate learned contraction
  critic. A perfect oracle gate held held-out F128 at `0.407` versus `5.372` ungated, demonstrating a
  real runtime-acceptance bottleneck. The learned critic reached AUROC `0.882`, but its preregistered
  false-safe rate was `28.3%` (required at most `15%`), so learned gated rollout evaluation was not
  permitted. The locked verdict was **ORACLE WORKS, LEARNED Q FAILS**.

The v4 result is therefore narrower than a stationary-learning claim: the controller learned useful
history-conditioned correction proposals, while neither the controller alone nor the learned critic
made their repeated execution reliably stable. Detailed artifacts live under `artifacts/v4_*`.

## Interpretation

The mechanism claim is supported only if Phase A spans held-out teacher updates, frozen IFI improves
over iterations, IFI beats AR-no-feedback on held-out novelty, and correct residual evidence causally
beats every corruption. The generated report evaluates these conditions; a completed run is not by
itself a successful scientific result.
