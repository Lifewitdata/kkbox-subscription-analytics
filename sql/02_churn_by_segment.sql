-- Churn rate by plan, auto-renew and city segment.
WITH latest_txn AS (
    SELECT msno, payment_plan_days, is_auto_renew,
           ROW_NUMBER() OVER (PARTITION BY msno ORDER BY transaction_date DESC) AS rn
    FROM transactions_v2
),
subscriber_segments AS (
    SELECT t.msno, t.payment_plan_days, t.is_auto_renew, m.city, tr.is_churn
    FROM train_v2 tr
    LEFT JOIN latest_txn t ON tr.msno=t.msno AND t.rn=1
    LEFT JOIN members_v3 m ON tr.msno=m.msno
)
SELECT 'plan' AS segment_type, CAST(payment_plan_days AS TEXT) AS segment,
       COUNT(*) AS subscribers, AVG(1.0*is_churn) AS churn_rate
FROM subscriber_segments GROUP BY payment_plan_days
UNION ALL
SELECT 'auto_renew', CAST(is_auto_renew AS TEXT), COUNT(*), AVG(1.0*is_churn)
FROM subscriber_segments GROUP BY is_auto_renew
UNION ALL
SELECT 'city', CAST(city AS TEXT), COUNT(*), AVG(1.0*is_churn)
FROM subscriber_segments GROUP BY city
ORDER BY segment_type, churn_rate DESC;
