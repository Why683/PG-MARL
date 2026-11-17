# Provenance gaps found in the local snapshot

## Verified local historical evidence

- A-D raw folder structure is present.
- Historical source-only v7-v14 plans, controller logs and protocol audits are
  present.
- Historical v8 final evaluation contains 19 D trials but only 152 windows
  (eight windows per trial), not the 5,700-window final manuscript protocol.
- Historical five-seed v8 checkpoint hashes and per-seed metrics are preserved
  in the companion archive; the actual checkpoint files are not present on the
  local workstation.
- A new A-D manifest was generated from the local folder tree using the final
  300-window-per-complete-trial rule. It contains 18,000 A/B/C rows and 5,700
  D rows and passes the source-only/disjointness audit. This verifies counts
  and access boundaries only; it is not evidence for the reported metrics.

## Missing for an evidence-complete 11.16 public release

- the trajectory-E rows needed to complete the locally generated A-D manifest;
- trajectory-E raw/processed records mapped to the manuscript's four complete
  five-state groups;
- all final five-seed teacher and student checkpoint files;
- per-window A/B/C/D/E logits and predictions for every reported method;
- run-level confusion matrices and metrics behind Tables 6, 7 and 9;
- source-only screening/ranking records for each baseline and PG-MARL;
- immutable configuration-freeze, E-acquisition and first-access timestamps;
- the exact numerical tooth-count/order record referenced by Section 3.1(C);
- numerical exports supporting all sensitivity, placebo, residual-decomposition,
  bootstrap and embedding tables.

The manuscript-stated values copied into `manuscript_claims/` must not be
represented as regenerated results until these artifacts are exported from the
server and pass the release verifier.
