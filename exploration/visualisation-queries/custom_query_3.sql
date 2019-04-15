-- custom SQL query for Tableau dashboard
-- visualisation: brand transaction distribution
SELECT
  num_transactions_binned,
  COUNT(brand) AS num_brand
FROM (
  SELECT
    *,
    CASE
      WHEN num_transactions = 1 THEN "1"
      WHEN num_transactions > 1 AND num_transactions <= 10 THEN "2 - 10"
      WHEN num_transactions > 10 AND num_transactions <= 50 THEN "11 - 50"
      WHEN num_transactions > 50 AND num_transactions <= 100 THEN "51 - 100"
      WHEN num_transactions > 100 AND num_transactions <= 250 THEN "101 - 250"
      WHEN num_transactions > 250 AND num_transactions <= 500 THEN "251 - 500"
      WHEN num_transactions > 500 AND num_transactions <= 1000 THEN "501 - 1000"
      WHEN num_transactions > 1000 AND num_transactions <= 2500 THEN "1001 - 2500"
      WHEN num_transactions > 2500 AND num_transactions <= 5000 THEN "2501 - 5000"
      WHEN num_transactions > 5000 AND num_transactions <= 10000 THEN "5001 - 10000"
      WHEN num_transactions > 10000 AND num_transactions <= 25000 THEN "10001 - 25000"
      ELSE "> 25000"
    END AS num_transactions_binned
  FROM (
    SELECT
      brand,
      COUNT(*) AS num_transactions
    FROM
      `PROJECT.DATASET.transactions`
    GROUP BY
      brand ) )
GROUP BY
  num_transactions_binned