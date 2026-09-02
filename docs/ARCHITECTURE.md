# Architecture

WorldZero separates an unprivileged policy from a trusted evaluator and a fixed stochastic kernel.

```text
detached JSON observation
          │
          ▼
       policy ── primitive action ──► fixed kernel
                                      │ clocks, RNG, accounting,
                                      │ death, snapshots, replay
                                      ▼
                             typed LawFamily callback
                                      │ immutable view
                                      │ typed transition
                                      ▼
                         validation + atomic application
                                      │
                                      ▼
                    evaluator evidence + central scoring
```

## Fixed substrate

V1 families share one bounded grid, raw and rich resources, terrain, three portable module classes, one embodied agent, and the `MOVE`, `PICK`, `DROP`, `CONSUME`, and `WAIT` actions. A plugin cannot replace those primitives or redefine time, randomness, death, or accounting.

The kernel owns the global state-independent proposal envelope. It draws simulated time, channel, target, and acceptance randomness before checking current applicability. Matched active and control worlds therefore consume the same proposal stream.

## Law-family boundary

`worldzero.laws` exposes immutable records and the `LawFamily` lifecycle: sample hidden JSON state from named draws; declare channels; derive evaluator state; propose typed transitions; project schema-checked public observations; define controls; emit standardized evidence; and declare bounded calibration cases.

Plugins never receive the mutable `World`. The kernel validates bounds, declared capabilities, finite values, material/energy deltas, and atomicity before applying any transition. Plugin-private persisted state is finite JSON.

## Registry and trust

Exact lowercase namespaced IDs resolve through a deterministic registry. Built-ins register directly. Installed community distributions advertise entry points in `worldzero.law_families`; listing reads metadata without importing implementations, and selection imports only the requested exact entry point.

Community plugins are trusted in-process code, not sandboxed extensions. They require the explicit `--experimental-family` flag. Official execution additionally authenticates descriptor, implementation, calibration, origin, and release status against the bundled registry.

## Observations and evidence

Policies receive a JSON-roundtripped copy of local public state. Hidden parameters, family identity, evaluator events, proposal indices, control labels, and future randomness are forbidden. Family evidence records standardized stages and references; final decisions come from a central scoring profile, never from plugin diagnostics.

Controls include matched null, mechanism knockout, matter-preserving broken geometry, and retained state. The benchmark reports negative controls and both all-ancestor and eligible-only inheritance effects.

## Persistence and replay

Plugin-backed state-v3 snapshots and trace-v4 records freeze the exact descriptor, source and calibration fingerprints, channel order, hidden JSON instance, private state, scorer identity, observations, actions, transitions, accounting, and terminal digests. Replay resolves the same family and checks the trajectory exactly.

Compatibility adapters and frozen fixtures retain state-v2 and trace-v2/v3 behavior. Old result directories are never silently upgraded or overwritten.
