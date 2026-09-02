# Contributing to WorldZero

Thank you for helping improve WorldZero. Contributions may target the kernel, developer tools, documentation, or a new law family.

## Before opening a change

1. Search existing issues and explain the scientific or engineering problem the change addresses.
2. Keep the fixed v1 substrate and information boundary intact. A non-privileged policy receives detached JSON only and never the mutable world or evaluator state.
3. Add tests before or with implementation changes. Do not relax tests, thresholds, controls, or denominators to obtain a preferred research outcome.
4. Use a new run identity whenever source, configuration, family behavior, calibration, or scoring changes. Never overwrite recorded results.

For a law plugin, read [docs/CONTRIBUTING_LAWS.md](docs/CONTRIBUTING_LAWS.md) and begin with [the example package](examples/community_law_plugin).

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m worldzero check-math --samples 768
```

Run the mathematical gate after changing a stochastic law, event channel, or simulated clock. All changes must preserve accounting invariants, absorbing death, empty successor memory, administrative censoring, matched negative controls, and exact replay.

## Pull requests

Keep each change focused. Include:

- the behavior and motivation;
- tests and exact commands run;
- any persistence, scoring, calibration, or compatibility impact;
- whether stochastic laws or clocks changed;
- confirmation that no paid model or external network call was needed; and
- documentation for user-visible behavior.

Official-family admission is a separate reviewed benchmark change. Passing `FamilyTestKit` does not make a community family official.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately as described in [SECURITY.md](SECURITY.md).
