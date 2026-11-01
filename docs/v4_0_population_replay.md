# IFI v4.0: Population-Replay Residual Learner

v4.0 is isolated from the v2/v3 teacher and functional-operator experiments. Its hidden
world is a bounded linear correction geometry, `R_next = R + B u`, and its action interface
is the identity. Controllers never receive `B`, a world identifier, trajectory age, optimizer
step, or positional information.

Each population slot is bootstrapped with two real random probe transitions. Training then
samples distinct slots uniformly, makes exactly one differentiable transition, optimizes a
scale-free contraction loss, detaches the transition, and stores it as real history. No loss
crosses a stored transition and no ordered rollout is optimized.

The three learned arms are query-history, current-only, and AR/no-fresh-residual. A fixed
linear rule calibrated only on training-family worlds and an analytic evaluation-only oracle
provide context. The frozen smoke evaluation runs to F64 on unseen same-family matrices and
on a locked condition-number 4–6 extrapolation bucket.

The locked GTX protocol is in `configs/v4_0_gtx_smoke.json`. Run:

```powershell
python -m ifi.v40_cli prepare --config configs/v4_0_gtx_smoke.json
python -m ifi.v40_cli throughput --config configs/v4_0_gtx_smoke.json
python -m ifi.v40_cli run --config configs/v4_0_gtx_smoke.json
```

At step 2,000, a preregistered gate requires query-history median q below 0.98 and median
F8/F0 below 0.90. Failure terminates the smoke without architecture changes. Passing the gate
continues the one seed to 12,000 local updates and triggers the complete frozen evaluation.
