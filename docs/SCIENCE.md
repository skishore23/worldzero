# Scientific specification

## Question

Can a finite-lived policy discover a useful environmental arrangement, let it perform work without continuing decisions, and leave a controlled advantage for a fresh successor?

This implementation tests whether that **opportunity and its causal measurement** exist. Scripted-control success is not evidence of LLM emergence.

## State and clocks

The state consists of the resource field, three portable component locations, terrain, a binary resource regime, a living individual's energy/age/private state, and evaluator-only metadata. Natural lifespan and action completion are scheduled boundaries; metabolism is continuous. Thinking takes positive simulated time. Observations are captured before thought; an action can fail when the world changes before completion. Real HTTP latency does not change simulated time.

Death by zero energy or lifespan terminates the individual. Acquired memory is cleared. A carried component is deposited locally, not deleted. Retirement dissipates residual bodily energy explicitly. A successor is a new policy instance with empty private state. The evaluator's archived logs never cross the observation interface.

## Stochastic law

Let N be the number of cells, F the set of source-capable cells, and rates s,d0,d1,c,dm,v denote source, raw decay, rich relaxation, conversion, module decay, and regime switching.

The total rate of *proposed* events is the constant

    Lambda = |F| s + N(d0 + d1 + c) + 3 dm + v.

Waiting times are Exp(Lambda). A channel/site is sampled proportional to its fixed proposal rate. Its local precondition decides whether the proposal is accepted. Ineligible proposals leave the state unchanged.

For a state function f, environmental dynamics therefore have generator

    L f(x) = sum_j lambda_j * 1[precondition_j(x)] * (f(T_j(x)) - f(x)),

with an additional acceptance multiplier for seasonal resource supply. This is uniformization/thinning, not a synchronous cellular automaton. The bounded finite world makes a global proposal-rate bound straightforward. It is less efficient than state-dependent methods in a sparse world but unusually convenient for exact shared-randomness counterfactuals.

The pending proposal is retained across observation and action boundaries. Cloning copies its time, channel, site, rejection uniform, and future RNG state. Turning off a mechanism does not change the proposal distribution. **Proposed** shocks remain the same; accepted events can differ because the intervention changes preconditions. That distinction is essential.

Method reference: Beentjes and Baker, *Uniformization techniques for stochastic simulation of chemical reaction networks*, J. Chem. Phys. 150, 154107 (2019), DOI 10.1063/1.5081043. Open manuscript: https://arxiv.org/abs/1811.00948 .

## Energy and material laws

The world is an OPEN system. Source events import material and raw-resource energy. Conversion captures additional external energy; adjacency does not create energy from nothing. Metabolism, cognition, decay, and retirement dissipate energy. Eating transfers resource energy into the individual and transfers its material out to an explicit waste tally.

Every run checks:

    initial energy + imported energy - dissipated energy = current stored energy
    initial material + imported material - removed material = current material

These are bookkeeping units, not a claim of thermodynamic realism. Shannon entropy is not thermodynamic entropy, and the environment does not reward agents for maximizing entropy.

## Hidden and visible information

Hidden: active pair, exact event rates, future proposals, outside-view contents, conversion counters, global causal field, evaluator interventions.

Visible: local object tokens, primitive portability and consumability affordances, local terrain category, position/bounds, energy/age, last action outcome, private note, and available actions/costs. Consumable energy values must be learned from outcomes. Visible tokens, layout, physics parameters, and trajectory RNG have separately derived seeds. Layout does not encode the active pair by placing it specially.

The runner serializes the observation before calling a policy. Non-privileged policies receive a plain detached JSON dictionary, never World. The informed control alone receives the active pair's visible labels at initialization, and is labeled accordingly. It is not an exact optimal-policy bound.

## Construction is not synonymous with discovery

A functional assembly is recorded when a physical DROP makes the active pair satisfy the local geometry. This is a mechanical proxy, not a claim about intent. Random action can sometimes assemble a useful arrangement. The scripted experimenter contains a strong prior to try portable pairs and monitor changed consumable outcomes. Its success validates a search/control opportunity, not spontaneous technological concepts.

For an LLM discovery claim, add blinded trace audit, appropriate learned/heuristic controls, held-out laws, deliberate reconstruction, and replication across model seeds. Public notes are evidence of reported hypotheses, not privileged access to private reasoning or causal proof by themselves.

## Inheritance interventions

Each completed ancestral world yields three branches, including worlds with no retained motif:

1. Retain the mechanism.
2. Disable conversion while retaining the same geometry (primary mechanism knockout).
3. Relocate one component, preserving matter (secondary geometry intervention).

The creator is absent for 20 time units. Counterfactual branches share all proposed random events. Record output during this interval. Then match every branch's resource array to the no-mechanism branch before birth. This removes the extra food stock as a mediator. Birth energy and the fixed home position are identical and do not depend on hidden motif location. Every successor is a new policy instance.

The primary estimand is the controlled effect of **continued inherited functionality after stock matching**, averaged over all completed ancestors. The eligible-only effect conditions on a retained motif and is reported separately. It must not be substituted for the all-ancestor estimate.

Because geometry's only functional effect in this kernel is the conversion channel, mechanism knockout and physical separation can yield identical successor paths. They are implementation cross-checks, not independent experimental replications.

## Statistics

Report sample size, completed and censored episodes, survival, assembly, retention, and decisions. Survival means reaching the precise lifespan boundary with no prior viability failure. Administrative call/decision exhaustion is censoring, not death and not survival. A censored candidate cannot pass the mechanical screen. Transport failures abort the cell and remain explicit.

Paired inheritance effects use the same source world and coupled randomness. The packaged 95% intervals are nonparametric bootstrap intervals over world seeds (4,000 fixed-seed bootstrap draws), not universal guarantees or a multiple-comparison-corrected family of claims. Report unconditional and conditional denominators. Wilson intervals accompany survival proportions.

## Numerical verification independent of the policy

1. With only source events, an empty fertile site contains raw resource at time T with probability 1-exp(-sT).
2. With RAW <-> RICH rates gamma and delta, starting RAW, the probability of RICH at T is gamma/(gamma+delta) * [1-exp(-(gamma+delta)T)].

`check-math` measures those probabilities using the production engine, not a second implementation of the formula. Snapshot/action replay and time-partition invariance are separate checks.

## Scope boundary

The model's engineered physics and controller priors define what is testable. This small law family will saturate; it does not support a claim of open-ended cultural accumulation. No experiment here demonstrates changing hierarchy, autonomous peer cooperation, emergence of intelligence from primitives, or transfer to software work.
