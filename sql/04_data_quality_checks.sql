-- Key, null, range and duplicate checks. Each query returns rows requiring review.
SELECT 'duplicate member key' AS check_name, msno AS offending_key
FROM members_v3 GROUP BY msno HAVING COUNT(*) > 1;

SELECT 'duplicate train key' AS check_name, msno AS offending_key
FROM train_v2 GROUP BY msno HAVING COUNT(*) > 1;

SELECT 'missing member key' AS check_name, CAST(NULL AS TEXT) AS offending_key
FROM members_v3 WHERE msno IS NULL;

SELECT 'invalid churn label' AS check_name, msno AS offending_key
FROM train_v2 WHERE is_churn NOT IN (0,1) OR is_churn IS NULL;

SELECT 'negative listening seconds' AS check_name, msno AS offending_key
FROM user_logs_v2 WHERE total_secs < 0 OR total_secs IS NULL;

SELECT 'invalid payment amount' AS check_name, msno AS offending_key
FROM transactions_v2 WHERE actual_amount_paid < 0 OR actual_amount_paid IS NULL;
