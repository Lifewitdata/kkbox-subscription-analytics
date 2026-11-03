# Stakeholder brief — decision framework

## Executive position

This repository contains synthetic demonstration data. It cannot support a real channel investment decision, a real CAC budget, or a claim that a particular subscriber segment is the highest-risk population in KKBox. The analysis demonstrates how those decisions should be made once validated first-party data are available.

## Acquisition channel decision

Use the notebook's cohort-level LTV and the SQL observed-revenue query as separate diagnostics. Do not rank channels using observed revenue per subscriber alone: acquisition cost, gross margin, refunds, channel mix, cohort maturity and retention must be aligned to the same acquisition cohort and measurement window.

For a real decision, calculate by channel:
- matured cohort size and retention at 1/3/6/12 months;
- contribution-margin LTV at 6/12 months;
- observed CAC from attributable campaign spend and acquired customers;
- LTV:CAC and payback period with uncertainty intervals.

Set the allowable CAC from a conservative lower confidence bound of contribution LTV divided by the required return multiple. The notebook's simple 3:1 ceiling is a scenario convention, not a validated policy.

## Retention action framework

Identify the high-risk segment only after leakage-safe out-of-time evaluation and calibration. Compare churn risk with segment size, expected contribution margin and intervention eligibility. Candidate actions to test include renewal reminders for non-auto-renew subscribers, payment recovery for failed/expired renewals, and re-engagement messaging for low-listening-activity subscribers.

Run a controlled experiment with a pre-defined primary outcome (e.g. 30-day retained-and-paying rate), guardrails (unsubscribe/complaint rate), and incremental contribution margin. Do not target users solely because of city or other sensitive proxies.

## What is required before a real recommendation

1. Replace synthetic data with authorised, privacy-reviewed customer and campaign data.
2. Correct the leakage in the legacy churn preprocessing and validate on a time-based hold-out.
3. Validate LTV on a later matured cohort; report decay-rate sensitivity and bootstrap intervals.
4. Join campaign spend to acquisitions at the same attribution definition.
5. Review calibration, segment stability and intervention uplift.

**Decision:** no channel should be recommended for investment from the current synthetic dataset. No CAC amount or highest-risk segment is claimed.
