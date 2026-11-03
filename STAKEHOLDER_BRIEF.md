# Stakeholder brief — KKBox subscription analytics

## Executive position

This repository demonstrates a reproducible analytics workflow on synthetic KKBox-schema data. It cannot support a real channel investment decision, a real CAC budget, real customer targeting or a claim that a particular retention action reduces churn.

## Churn-risk analysis

The model's output is a prioritisation signal for a demonstration only. The pipeline uses train-only preprocessing and a chronological signup-date hold-out, but the synthetic source data do not provide a production-grade point-in-time feature snapshot. A real deployment requires event-time cutoffs, an observation window and repeated out-of-time evaluation.

Use risk with subscriber eligibility, contribution margin, contact policy and treatment capacity. Do not target customers solely because of city or demographic proxies.

## Acquisition channel decisions

Use the six-month observed LTV and its out-of-time cohort validation as separate diagnostics. It is not the previous 24-month decay extrapolation validation. Do not rank channels using revenue per subscriber alone: acquisition cost, contribution margin, refunds, cohort maturity, attribution and channel mix must be aligned to the same acquisition cohort and measurement window.

For a real decision, calculate by channel:
- matured cohort size and retention at 1/3/6/12 months;
- contribution-margin LTV at 6/12 months;
- observed CAC from attributable campaign spend and acquired customers;
- LTV:CAC and payback period with uncertainty intervals.

Set allowable CAC from a conservative lower bound of contribution LTV divided by the required return multiple. The 3:1 convention is a scenario assumption, not a validated policy.

## Retention intervention evaluation

The repository includes an intention-to-treat evaluator that requires random assignment and a pre-defined outcome such as 30-day retained-and-paying status. It does not contain offer assignment, redemption, or post-offer outcome data. The committed placebo check is only a design demonstration using historical labels; it is not an experiment and cannot establish uplift.

A real test should:
- randomize eligible subscribers to offer and control before messaging;
- define the primary outcome and follow-up window before launch;
- analyse by assignment, not offer redemption;
- track unsubscribe/complaint rate, contact cost, incremental contribution margin and retention;
- pre-specify sample size, stopping rules and subgroup analysis.

Candidate actions to test include renewal reminders for non-auto-renew subscribers, payment recovery for failed/expired renewals and re-engagement messaging for low-listening-activity subscribers. Do not describe descriptive segment differences as evidence that an action works.

## What is required before a real recommendation

1. Replace synthetic data with authorised first-party events and campaign spend.
2. Create a point-in-time feature contract and repeated out-of-time churn validation.
3. Validate 6/12-month contribution LTV on later matured cohorts and quantify attribution, margin and decay uncertainty.
4. Join actual acquisition spend to acquired customers with an agreed attribution definition.
5. Run a randomized retention experiment and evaluate intention-to-treat impact.

**Decision:** no channel or retention action should be recommended for real investment from this synthetic dataset.
