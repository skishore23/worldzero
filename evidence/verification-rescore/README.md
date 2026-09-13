# Keeping credit for a demonstrated reconstruction

The original [verification study](../verification-study/README.md) is preserved unchanged. This is a post-hoc comparison of two scoring versions on its existing trajectories, not a new experiment.

## The correction

`levels-v2` requires `retained_or_reconstructed`, which the built-in families derive from terminal functionality. A later module failure can therefore cancel credit for an earlier verified reconstruction. `levels-v3` uses the validated, event-linked reconstruction witness as the historical evidence. Terminal retention remains recorded, and Level 5 still requires eligible controlled successor outcomes. Level 3 still requires survival and a supported finding; Level 4 additionally requires linked benefit.

## Same trajectories, different interpretation

| Agent | Active worlds | Survival | Verification | Level 3: v2 → v3 | Level 4: v2 → v3 |
|---|---:|---:|---:|---:|---:|
| blind | 24 | 9 | 0 | 0 → 0 | 0 → 0 |
| forager | 24 | 14 | 0 | 0 → 0 | 0 → 0 |
| retain | 24 | 13 | 0 | 0 → 0 | 0 → 0 |
| verify | 24 | 11 | 12 | 8 → 10 | 8 → 10 |

All 192 original trace digests and old scores were checked. Exactly 2 episode scores changed:

| Family | Seed | Agent | v2 | v3 | Original trace |
|---|---:|---|---:|---:|---|
| worldzero:catalysis | 1792432103 | verify | 2 | 4 | [trace](../verification-study/traces/verify-worldzero-catalysis-active/1792432103.json.gz) |
| worldzero:delayed-transformation | 1792432103 | verify | 2 | 4 | [trace](../verification-study/traces/verify-worldzero-delayed-transformation-active/1792432103.json.gz) |

The verifier's Level 4 rate changes from 8/24 (33.3%) to 10/24 (41.7%). Its 12 verification witnesses and 11 survivors are unchanged. Two verified agents still fail the survival requirement. Null findings, all other agents' scores, and the original study's primary verification contrast are unchanged. These correlated episodes share a world seed; they are not independent replications.

## Versioning and reproduction

New benchmark manifests select `worldzero:levels-v3`. Standalone scoring helpers preserve their historical v2 default for compatibility; pass the manifest's `scoring_profile` explicitly. V3 refuses Boolean-only historical evidence: it requires the modern witness-bearing record. Existing frozen manifests still reject implementation drift; use their original checkout to resume reproduction.

The historical study runner remains explicitly pinned to v2. After regenerating its full traces with the study commands and the recorded Python/NumPy versions, run:

```bash
python -m scripts.verification_rescore \
  --trace-root runs/verification-study \
  --output runs/verification-rescore
```

The repository carries ten illustrative/diagnostic traces; this comparison requires the complete 192-trace run. It fails if any trace, original finding, episode outcome, or old score disagrees. [Results and scoring-source hashes](results.json) record the comparison. No model calls are made.
