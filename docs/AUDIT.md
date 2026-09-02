# Audit of the earlier prototype and changes in 0.2

The supplied `worldzero_lab_source.zip` was inspected. Its original five tests passed in this session. Passing those tests was not sufficient to justify all previous methodological descriptions.

## Corrected or bounded in this version

**Information parity.** The earlier `GenericExperimenterPolicy.decide(world)` inspected global grids, latent material classes, fertile terrain, and global rich-resource counts. The old forager also knew consumable latent identities. The new non-privileged interface receives only detached local JSON observations. The new forager learns returns from consumption. A separately labeled informed controller is allowed the active pair.

**Time and randomness.** The earlier action path charged cognition energy but did not represent a separate positive thinking duration. The revised engine schedules both. Repeated direct-method calls across action boundaries could change the sampled random path; the new state-independent proposal process preserves pending events and supports exact coupling across counterfactuals.

**Accounting.** The old enriched-resource transformation had no explicit external-energy ledger. The revised open-system ledger credits source and conversion input, records dissipation, preserves carried components at death, and asserts balance after every episode.

**Success classification.** The previous result used a one-time-unit tolerance for natural survival and could end a loop at a decision cap without explicitly identifying censoring. The revised system distinguishes starvation, lifespan, model-budget censoring, and infrastructure failure. Survival is determined by the actual terminal cause.

**Inheritance controls.** The earlier pair selected a successor birth site using motif proximity and used a separate lower starting energy. The revised default uses the same fixed home and same energy as the creator. It matches resource arrays after the idle interval and compares mechanism knockout as well as geometric ablation. All-ancestor and motif-conditional effects are both reported.

**Model output validity.** Missing/unknown actions can no longer become unflagged WAIT actions. Invalid responses are counted. Network/HTTP failures abort rather than masquerade as agent behavior. Calls are capped. Remote requests require explicit authorization in configuration; API keys come from an environment variable and are not stored in traces.

**Run provenance.** The old run ledger did not bind source code, model parameters, prompt, and configuration tightly enough. The new SQLite ledger binds them and rejects silent redefinition or overwritten committed cells. External inference is not exactly-once: restarting an uncommitted episode can repeat paid calls, so incomplete-cell retry is explicit.

**Novelty and interpretation.** Opaque labels do not make a hand-written interaction law emergent physics. A scripted pair search does not establish LLM invention. An inherited advantage does not establish open-ended culture. The reports now separate these claims.

## Deliberate differences from the prior world

Resources and portable components use separate layers, so moving a component cannot secretly alter food availability merely by occupying a cell. This makes the causal intervention cleaner but changes the benchmark. The new rates, observations, policies, and counterfactual protocol are not numerically comparable to earlier published percentages. The development pilot and revised pressure sweep are retained in `evidence/`.

The first easy development setting had a survival ceiling. That was recorded, not presented as an inheritance success. The pressure setting was calibrated using development seeds 100..107; the full validation used a new 64-seed local test split. Future studies should freeze a new split because this evidence release publishes its seeds.
