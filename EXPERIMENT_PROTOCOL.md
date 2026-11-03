# Prospective retention experiment protocol

This repository contains no actual retention campaign, randomized offer assignment or follow-up outcomes. This protocol prepares a real test; it does not claim the test has run.

## Proposed primary question

Among eligible subscribers whose renewal is approaching and who have not already churned, does a pre-defined renewal-support offer increase 30-day retained-and-paying rate versus business-as-usual communication?

## Before randomization

1. Obtain approval from business, privacy/legal and customer-contact policy owners.
2. Define the eligible population and exclusions before viewing outcomes.
3. Freeze an experiment start date, a single 30-day outcome window and the primary metric.
4. Choose one offer and one message. Do not mix different incentives or treatments within an arm.
5. Use a historical real baseline retention rate, define a minimum detectable absolute uplift and run the sample-size planner. The baseline and MDE are assumptions until approved.

## Sample-size planning

~~~bash
python scripts/plan_retention_experiment.py --baseline-retention 0.80 --minimum-detectable-uplift 0.03 --alpha 0.05 --power 0.80 --nonresponse-rate 0.05
~~~

The example values above are illustrative inputs only; replace them with a business-approved real baseline and MDE.

## Create assignment

Prepare an authorised eligibility CSV with one row per subscriber and a pre-defined stratum column. Do not create strata using post-assignment outcomes.

Set a private randomization seed in the environment; never commit it:

~~~bash
# PowerShell
$env:RETENTION_EXPERIMENT_SEED = "a-long-random-secret-held-outside-git"

# macOS/Linux
export RETENTION_EXPERIMENT_SEED="a-long-random-secret-held-outside-git"

python scripts/prepare_retention_experiment.py --input eligible_subscribers.csv --output assigned_subscribers.csv
~~~

The script applies deterministic HMAC ranking within each stratum and assigns near-equal counts to treatment and control. Preserve the output and key securely so assignment can be audited. Operational delivery of the offer is deliberately not automated by this repository.

## Record delivery and outcomes

Join the frozen assignment to contact-platform delivery/redeem events and the billing system's 30-day status. Use the columns in experiment_template.csv. Keep assigned_treatment as the original assignment even if an offer was not delivered or redeemed.

- offer_sent: delivery confirmation, not assignment.
- offer_redeemed: redemption flag, not assignment.
- retained_30d: pre-defined retained-and-paying primary outcome.
- unsubscribed_30d: customer-contact safety guardrail.
- contribution_margin_30d: realised incremental margin after discount and contact costs, using an agreed finance definition.

## Analyse intention-to-treat

~~~bash
python scripts/evaluate_intervention.py --input completed_experiment.csv --output results/intervention/experiment_itt.json
~~~

Report arm sizes, retention rates, absolute/relative uplift, confidence interval and p-value, plus unsubscribe and contribution-margin guardrails when supplied. Analyse original assignment (intention-to-treat), not redemption or successful delivery. Do not stop early because a result looks favourable; use the pre-specified stopping date and multiplicity policy.

## Interpretation

This project currently has no real campaign or outcome data. The placebo result is a code/design check on historical synthetic labels, not a treatment effect. No causal retention claim is justified until a properly governed randomized experiment has run and its outcomes have been analysed.
