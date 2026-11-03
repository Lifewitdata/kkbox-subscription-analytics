-- SQLite-compatible cohort retention by signup month and age month.
WITH signup AS (
    SELECT msno, substr(CAST(registration_init_time AS TEXT), 1, 6) AS cohort_month
    FROM members_v3
),
activity AS (
    SELECT DISTINCT msno,
           CAST(strftime('%Y', date(substr(CAST(date AS TEXT),1,4)||'-'||
             substr(CAST(date AS TEXT),5,2)||'-'||substr(CAST(date AS TEXT),7,2))) AS INTEGER)*12
           + CAST(strftime('%m', date(substr(CAST(date AS TEXT),1,4)||'-'||
             substr(CAST(date AS TEXT),5,2)||'-'||substr(CAST(date AS TEXT),7,2))) AS INTEGER) AS activity_month_num
    FROM user_logs_v2
),
cohort_activity AS (
    SELECT s.cohort_month, a.msno,
           a.activity_month_num -
           (CAST(substr(s.cohort_month,1,4) AS INTEGER)*12 +
            CAST(substr(s.cohort_month,5,2) AS INTEGER)) AS age_month
    FROM signup s JOIN activity a USING (msno)
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT msno) AS cohort_users
    FROM signup GROUP BY cohort_month
)
SELECT ca.cohort_month, ca.age_month,
       COUNT(DISTINCT ca.msno) AS active_users,
       cs.cohort_users,
       1.0 * COUNT(DISTINCT ca.msno) / cs.cohort_users AS retention_rate
FROM cohort_activity ca JOIN cohort_sizes cs USING (cohort_month)
WHERE ca.age_month >= 0
GROUP BY ca.cohort_month, ca.age_month, cs.cohort_users
ORDER BY ca.cohort_month, ca.age_month;
