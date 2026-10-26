# Working in this repository

Read README.md, METHODS.md, CONTRIBUTING.md and matrix/README.md first.

- Back measured claims with receipts. Preserve unknowns and failed results.
- Edit matrix/capabilities.json, then regenerate views with
  python tools/build_matrix.py. Do not hand-edit generated views.
- Preserve existing receipts and record the probe revision and run configuration.
- Follow METHODS.md before probing: use a quiet, thermally settled machine,
  run one benchmark at a time, and record power mode. Do not stop unrelated
  processes or change system settings without the owner's agreement.
- Publish only approved hardware metadata and sanitized diagnostics.
  Exclude credentials, personal identifiers, private paths and raw logs.
- Run python tools/build_matrix.py --check, the regression tests
  (`python -m unittest discover -s tests`), and compile-check all probes:
  `python -m py_compile probes/_common.py probes/lane_sweep.py probes/integer_lanes.py
  probes/geometry_sweep.py tools/phase0a_capture_spike.py tools/inspect_gpu_trace.py`. Changes to probe behavior require correctness checks on
  suitable hardware.
