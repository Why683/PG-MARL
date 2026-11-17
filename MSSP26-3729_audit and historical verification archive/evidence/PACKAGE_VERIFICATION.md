# Package verification record

Packaging date: 2026-11-16 (Asia/Shanghai).

## Completed locally

- The manuscript-aligned Python sources pass `compileall` syntax checking.
- The fixed-manifest builder and source-only protocol verifier were executed
  against the locally available A-D folder tree.
- The audit produced 18,000 source rows and 5,700 target rows: each of A/B/C
  contains 3,000 training, 1,500 validation and 1,500 test windows; D contains
  5,700 `target_test` windows.
- Source trials/windows are disjoint and D is not registered as a training or
  validation domain.
- A SHA-256 ledger is generated for every release file.

## Not completed locally

- The synthetic tensor tests require PyTorch and PyTorch Geometric. The
  isolated packaging runtime did not contain PyTorch, so these tests were not
  executed on this workstation. Run `python -m unittest discover -s tests -v`
  in the server `py38` environment after installing `requirements.txt`.
- No claim of numerical reproduction is made because the local snapshot lacks
  trajectory E, the five final checkpoints and the final A-E prediction files.

The manifest audit verifies the stated access boundary and sample counts; it
does not establish the reported classification performance.
