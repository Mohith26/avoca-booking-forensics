# Booking Forensics: lost-booking blame + judge-calibrated replay lift (concept for Avoca)

**Live demo:** https://mohith26.github.io/avoca-booking-forensics/

An unofficial concept screen for an AI call-handling dashboard in home services (HVAC, plumbing, electrical). It answers two questions a deployed engineer asks when tuning an agent for one customer:

1. **Where exactly did each lost booking go wrong?** Not "why didn't it book" at the call level, but *which turn* lost it, how much each failure mode is worth, and whether it is fixable in prompt/policy config.
2. **If I change the prompt or policy, how much will booking rate move, and how sure am I, before any live caller hears it?**

The screen is presented as a new **Responder → Booking Forensics** tab, styled after the dashboard as shown in Avoca's public help-center screenshots (Instrument Sans, the dashboard's own color tokens, dark sidebar, Lucide icons, Coach-style cards, tables and call details). Switch teams from the team switcher at the top of the sidebar.

Unofficial concept demo by Mohith Gajjela. Not affiliated with Avoca. All data is synthetic (fictional businesses, callers and addresses).

## Why this and not what already exists

Public help docs describe Top Unbooked Reasons (call-level reasons), Drop-off Analysis (where in the journey a call ended), Oversight evals and annotations, Simulation Testing with a regression suite built from real calls (pass/fail per case), and live A/B experiments. The gap this demo targets sits between the last two:

* A regression suite tells you whether hand-written success criteria pass. It does not tell you the **booking-rate lift** of a change across your call mix, or how many currently-booked calls the change would break.
* A live A/B test answers that, but it spends real callers, takes weeks at one customer's volume, and you have to guess an effect size to size it.
* Replaying past calls with an LLM judge is the obvious middle step, but a judge is biased in ways you cannot see from its own output. In this demo the judge is confidently wrong about one of the three changes (its 95% interval covers the truth 0-3% of the time).

The fix is to treat the judge as a **prediction**, not a label, and correct it with a small number of human reviews using **prediction-powered inference (PPI++)**. The result is an interval that stays valid however biased the judge is, and is tighter than using the human reviews alone.

## How it works

**1. Turn tagging.** A small rule-based tagger marks agent dialogue acts (deflected price, quoted fee, no alternative slot, routine visit for an urgent issue, address re-ask, booking-tool error...) and caller cues (urgent situation, price question, comparison shopping, hesitation, exit). In production this would be an Oversight annotation or a small classifier.

**2. Prefix value model.** An L2-regularised logistic regression (fit by IRLS, in the browser) predicts `V(prefix) = P(booked | everything said so far)` from tagged features, with customer fixed effects. It is trained on every prefix of every lead (about 17k rows).

**3. Blame localization.** For each lost call, each agent turn gets the change in `V` from just before the turn to just after the caller's reaction. Drops at benign turns (a generic "let me call you back" after the address was taken) are evidence about an earlier problem, so their credit is carried back to the open grievance: the latest risky turn the caller reacted badly to. The blamed turn is the one with the most credit; its act defines the failure mode. On the synthetic set this recovers the planted cause on ~92% of agent-caused losses versus ~74% for "blame the last agent turn before hang-up", and the gap is largest for callers who linger a few turns before leaving.

**4. Failure modes.** Lost leads are grouped by mode with job value at stake and whether the mode is fixable in config (prompt/policy), an integration problem, or not agent-side.

**5. Replay design.** For a proposed change, leads are stratified (booked + one stratum per failure mode). A 30% pilot is allocated proportionally; the rest of the replay budget uses Neyman allocation. A replay that takes the same dialogue acts as the original call keeps its observed outcome. Only *diverged* replays need judging, and human review goes only to a random subset of those.

**6. Estimation.** Per stratum, the diverged part is estimated three ways and combined with exact stratum weights:
* judge only: `mean(f)` (precise, biased)
* human only: `mean(y)` on reviewed replays (unbiased, wide)
* **PPI++**: `λ·mean(f_unreviewed) + mean(y − λ·f)_reviewed` with power-tuned `λ` ([Angelopoulos et al., 2023](https://arxiv.org/abs/2311.01453))

The screen reports the projected booking rate, lift, recovered lost calls, **booked calls the change breaks**, booked job value per month, a ship / hold / regression verdict, and an A/B sample size powered on the lower bound of the offline estimate.

**7. Interval honesty check.** Because the data is simulated, the true post-change booking rate is known. "Check interval honesty" reruns 200 independent replay designs and reports how often each method's 95% interval contains the truth.

## The synthetic world

`data/generate.py` (stdlib only, seed `20261030`) simulates three fictional businesses over October 2026 (~1.9k leads). Each caller has hidden traits (price sensitivity, urgency, flexibility, patience) and fixed random draws; agent policy chooses acts at decision points (triage, price question, address capture, scheduling, booking tool, close), each with a hazard that the caller leaves. Some callers linger 1-2 turns before ending with a generic excuse. Replaying a call under a new policy reuses the same caller and draws (common random numbers), so the counterfactual outcome is exact inside the simulation. A simulated LLM judge scores replays with noise and an optimistic bias toward the "fixed" behaviour.

Three candidate changes ship with the data:
* **Quote the diagnostic fee when asked** recovers deflected price questions but makes some callers who would have booked balk at a stated fee. Net effect is small; the judge says it is a clear win.
* **Offer the two nearest open windows + waitlist** recovers "no alternative offered" losses with essentially no regressions.
* **Same-day emergency escalation** recovers urgent callers triaged as routine; a few non-emergencies get routed to the emergency fee.

## Run it

```
python3 data/generate.py             # regenerate data (deterministic)
node tests/core.test.js              # 15 tests: tagger, IRLS, quantiles, A/B sizing, PPI++ bias/coverage, localization accuracy, design determinism, per-candidate coverage
python3 -m unittest discover -s tests   # generator tests: determinism, common-random-number replays, fatal-turn marking
python3 -m http.server               # then open http://localhost:8000
```

No build step; `core.js` runs unchanged in Node and the browser.

## Honest limitations

* **Synthetic data, templated text.** Transcripts are generated from templates with variation, so the rule tagger is near-perfect here. Real transcripts need a learned tagger, and localization accuracy would be lower. The ~92% figure measures the attribution logic, not real-world accuracy.
* **Replays are not real.** The counterfactual is exact only inside the simulator. On a real system, replays are LLM-simulated callers, which adds a sim-to-real gap that no amount of statistics on the replay sample removes. PPI corrects the judge relative to human reviewers of replays, not relative to real callers.
* **Divergence by act sequence** assumes a replay that takes the same dialogue acts ends the same way. Real agents are stochastic; in practice you would replay unchanged calls a few times or treat them as a separate stratum.
* **The value model is correlational.** Blame is "the turn where predicted booking probability fell", not a causal effect. The replay step is where causal claims are made.
* **Variance floors** keep intervals from collapsing when a stratum's reviews are all one outcome; they make the human-only interval somewhat conservative.
* Job value per lead is an average ticket by service type, not revenue.
