-- Revenue and proxy LTV by registration channel.
-- This query is revenue per labelled/acquired subscriber, not a forward forecast.
WITH revenue AS (
    SELECT msno, SUM(actual_amount_paid) AS observed_revenue
    FROM transactions_v2 GROUP BY msno
),
subscriber_value AS (
    SELECT m.registered_via AS channel,
           m.msno,
           COALESCE(r.observed_revenue, 0) AS observed_revenue
    FROM members_v3 m LEFT JOIN revenue r USING (msno)
)
SELECT channel,
       COUNT(*) AS subscribers,
       SUM(observed_revenue) AS observed_revenue,
       AVG(observed_revenue) AS observed_revenue_per_subscriber
FROM subscriber_value
GROUP BY channel
ORDER BY observed_revenue_per_subscriber DESC;
