# Contributing a law family

WorldZero law families add hidden causal mechanisms to one fixed stochastic
substrate. A plugin reuses the kernel's clocks, named randomness, actions,
accounting, controls, snapshots, replay, scoring, and visualization. It does
not receive the mutable world or define its own benchmark decision rule.

## Trust boundary

Plugins are trusted, in-process Python. Installing or loading a community
plugin grants it the same local process permissions as WorldZero. V1 does not
sandbox plugins. The CLI prints this warning before importing a selected
community entry point, and community families require
`--experimental-family`.

Policies remain unprivileged. They receive detached JSON observations only.
A plugin must never expose hidden parameters, family identity, evaluator
events, control labels, proposal indices, future randomness, or global
mechanism state through `project_public()`.

## Lifecycle

For each exact family ID, the evaluator follows this sequence:

1. Resolve one entry point and freeze its descriptor, source fingerprint, and
   calibration-suite fingerprint.
2. Call `sample()` with kernel-owned named draws.
3. Freeze the state-independent envelope returned by `channels()`.
4. Call `derive()` on detached immutable substrate views.
5. Offer matching proposal draws to `apply_proposal()` and kernel proposal
   filters; validate and atomically apply typed operations.
6. Validate `project_public()` against the descriptor's closed observation
   schema before JSON-roundtripping it to a policy.
7. Build matched controls from `controls()` and `intervene()` while the kernel
   owns cloning, clocks, randomness, accounting, death, and successor state.
8. Convert evaluator traces to standardized `FamilyEvidence`; central scoring
   ignores plugin-specific success claims.
9. Persist exact family identity in state-v3 and trace-v4 and resolve that same
   identity for replay.

The stable public imports live under `worldzero.laws`: `LawFamily`, the typed
records, `LawRegistry`, `FamilyTestKit`, and registry helpers. Plugins should
not import `World`, `worldzero.kernel` internals, or built-in family helpers.

## Minimal package

See [`examples/community_law_plugin`](../examples/community_law_plugin). Its
entry point is:

```toml
[project.entry-points."worldzero.law_families"]
"example_org:preserver" = "worldzero_example_law:family"
```

The entry-point name and `FamilyDescriptor.family_id` must be the same exact,
lower-case namespaced ID. The target returns one `LawFamily` instance or a
zero-argument factory that creates one. Resolution is exact; aliases and fuzzy
matching are not supported.

After installing the package locally:

```bash
worldzero laws list
worldzero laws inspect example_org:preserver --experimental-family
worldzero laws validate example_org:preserver --experimental-family
```

`laws list` reads package metadata but does not import community
implementations. `inspect` and `validate` import only the selected entry point.
A selected plugin error stops validation; WorldZero never substitutes another
family.

## Typed state and transitions

All persistence-bound values must be finite JSON. `FamilyInstance` separates
hidden parameters from bounded plugin-private state. Both are copied and
recursively frozen. Use only kernel-supplied named draws from `SampleContext`;
do not use `random`, `numpy.random`, clocks, files, services, or ambient global
state to determine behavior.

Channels declare a fixed envelope rate, target domain, and draw requirements.
The envelope must not depend on current substrate state. Preconditions belong
inside proposal callbacks, preserving matched proposal streams across active
and control worlds.

Return only the closed operations declared in `worldzero.laws` and include
exact material and energy deltas. The kernel validates a candidate transition
on copies and commits atomically. A callback exception or invalid transition
is a family-validation failure, not an invalid policy action, and never causes
fallback.

Private-state changes use `PrivateStateTransition`. Delayed behavior uses
kernel simulated time and `internal_deadline()`; wall-clock timers are not part
of simulator time.

## Observations

Declare a closed object schema in the descriptor. V1 supports objects, arrays,
null, booleans, integers, finite numbers, and strings. Every object must set
`additionalProperties = false`.

Projection may depend only on locally observable public substrate state. Test
active, null, knockout, and broken instances with identical public inputs.
Changing hidden parameters, private state, enablement, family identity, or
evaluator history must not change the projection when the public view is
unchanged.

## Controls and evidence

Every family declares `null`, `knockout`, `broken`, and `retained` controls.
Every arm declares the exact matching set `material_stock`, `proposal_stream`,
and `public_substrate`; empty or invented matching claims are rejected.
Null and knockout disable the mechanism without changing geometry, stocks,
private state, channel order, or proposal rates. Broken is deterministic and
matter-preserving and disrupts the required physical relation. Retained leaves
the ancestral state unchanged.

`evaluate()` returns standardized stage evidence and optional diagnostics. It
does not decide whether a benchmark passes. Central scoring owns thresholds,
coverage, censoring, invalid-action denominators, negative controls, and both
all-ancestor and eligible-only inheritance effects.

## Calibration and validation

Calibration cases must be bounded, deterministic, finite, and ordered. Their
complete metadata is fingerprinted. Community validation accepts only the
benchmark-owned `validator_contract` kind with `expected=True`, zero
tolerances, and exactly one `parameters.contract` selected from:
`deterministic_callbacks`, `transition_accounting`, `matched_controls`,
`observation_boundary`, `snapshot_replay`, or `lifecycle`. The validator
executes the named contract; unknown or unsupported claims fail instead of
being treated as evidence. Mechanism-specific analytic expectations still
belong in the plugin package's tests. Official admission replaces this limited
community catalog with the family's exhaustive benchmark-owned executable case
contract; missing, extra, reordered, metadata-drifted, throwing, or failing
official cases are rejected.

Run the reusable validator before submitting:

```python
from worldzero.laws import FamilyTestKit, installed_registry

report = FamilyTestKit(installed_registry()).validate(
    "example_org:preserver",
    seeds=range(16),
    include_calibration=True,
)
assert report["passed"], report["failures"]
```

The JSON report covers descriptor/fingerprint identity, deterministic sampling,
every callback's ambient/private RNG use, retained callback inputs and detached
input mutation, state-independent channels, recursive observation identity
leakage, all control arms and exact matching constraints, every finite declared
target plus boundary/midpoint acceptance draws, death/successor state, state-v3
roundtrip, trace-v4 replay, and executable calibration contracts. Hidden/private
projection probes independently re-derive each valid shape-preserving variant.
BROKEN controls are also exercised on bounded row, column, and diagonal
structural layouts.

Plugin-owned retention inspection is cycle-safe and covers mappings, sequences,
sets, dataclasses, named tuples, ordinary instance dictionaries, and declared
slots without evaluating arbitrary properties. Each callback scan is bounded to
4,096 unique nodes, depth 32, and 1 MiB of inspected storage. Exceeding any
limit is a structured validation failure, not a crash or a partial certificate.
Official families additionally run their exhaustive benchmark-owned calibration
suite.

Central score v3 resolves the exact registered implementation again and freezes
its descriptor, implementation and calibration fingerprints, origin,
official/experimental status, and scoring-profile identity. A supplied registry
may locate installed code, but it is never the official trust root: official
scoring independently authenticates every identity field against WorldZero's
bundled reviewed registry. Caller-authored allowlist rows cannot promote a
community family. A local experimental score requires an explicit experimental
flag and remains labeled experimental; historical score v1/v2 artifacts retain
their versioned compatibility behavior.

Also test the built wheel in a clean environment without network access, run
one deterministic episode, and replay its trace-v4. The example package's test
demonstrates that complete workflow.

## Versions and fingerprints

- Plugin API compatibility follows semantic versioning. A breaking public SDK
  change increments the API major version.
- A behavior, hidden-parameter, channel, control, evidence, or calibration
  change increments `family_version` and creates a new source fingerprint.
- Package releases increment `package_version` normally.
- A changed descriptor, implementation module, calibration tuple, channel
  order, scoring profile, or source revision creates a new run identity. Never
  reuse a committed result identity after such a change.

## Official admission checklist

An official-family proposal must include:

- a stable exact ID and documented scientific hypothesis;
- typed implementation using only the fixed v1 substrate and public SDK;
- deterministic named-draw sampling and a state-independent envelope;
- active, matched-null, knockout, broken, and retained tests;
- observation-blindness and detached-mutation tests;
- operation bounds and exact accounting tests;
- snapshot, replay, absorbing-death, empty-successor-memory, and matched-clock
  tests;
- bounded analytic/invariant calibration with benchmark-owned expectations;
- negative controls and standardized evidence, without family-owned pass rules;
- an offline built-wheel validation report and documentation;
- registry review of the exact descriptor, source fingerprint, calibration
  fingerprint, versions, and release status.

Admission is a reviewed benchmark change. Passing `FamilyTestKit` alone does
not make a community family official.
