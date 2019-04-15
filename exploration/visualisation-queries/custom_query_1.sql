-- custom SQL query in Tableau dashboard
-- visualisation: customer - transaction distribution
SELECT
  num_transactions_binned,
  COUNT(id) AS num_customers
FROM (
  SELECT
    *,
    CASE
      WHEN num_transactions = 1 THEN "1"
      WHEN num_transactions = 2 THEN "2"
      WHEN num_transactions > 2 AND num_transactions <= 5 THEN "3 - 5"
      WHEN num_transactions > 5 AND num_transactions <= 10 THEN "6 - 10"
      WHEN num_transactions > 10 AND num_transactions <= 20 THEN "11 - 20"
      WHEN num_transactions > 20 AND num_transactions <= 30 THEN "21 - 30"
      WHEN num_transactions > 30 AND num_transactions <= 50 THEN "31 - 50"
      WHEN num_transactions > 50 AND num_transactions <= 100 THEN "51 - 100"
      WHEN num_transactions > 100 AND num_transactions <= 250 THEN "101 - 250"
      WHEN num_transactions > 250 AND num_transactions <= 500 THEN "251 - 500"
      WHEN num_transactions > 500 AND num_transactions <= 1000 THEN "501 - 1000"
      WHEN num_transactions > 1000 AND num_transactions <= 2500 THEN "1001 - 2500"
      WHEN num_transactions > 2500 AND num_transactions <= 5000 THEN "2501 - 5000"
      WHEN num_transactions > 5000 AND num_transactions <= 10000 THEN "5001 - 10000"
      ELSE "> 10000"
    END AS num_transactions_binned
  FROM (
    SELECT
      id,
      COUNT(*) AS num_transactions
    FROM
      `PROJECT.DATASET.transactions`
    GROUP BY
      id ) )
GROUP BY
  num_transactions_binned