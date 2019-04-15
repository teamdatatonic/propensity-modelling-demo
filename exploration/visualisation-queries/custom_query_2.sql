-- custom SQL query for Tableau dashboard
-- visualisation: Average Order Value (excluding returned items)
SELECT
  AOV_binned,
  COUNT(id) AS num_customers
FROM (
  SELECT
    *,
    CASE
      WHEN AOV < 10 THEN "< 10"
      WHEN AOV >= 10 AND AOV <= 20 THEN "10 - 20"
      WHEN AOV > 20 AND AOV <= 30 THEN "20 - 30"
      WHEN AOV > 30 AND AOV <= 50 THEN "30 - 50"
      WHEN AOV > 50 AND AOV <= 100 THEN "50 - 100"
      WHEN AOV > 100 AND AOV <= 250 THEN "100 - 250"
      WHEN AOV > 250 AND AOV <= 500 THEN "250 - 500"
      WHEN AOV > 500 AND AOV <= 1000 THEN "500 - 1000"
      WHEN AOV > 1000 AND AOV <= 5000 THEN "1000 - 5000"
      WHEN AOV > 5000 AND AOV <= 10000 THEN "5000 - 10000"
      ELSE ">10000"
    END AS AOV_binned
  FROM (
    SELECT
      id,
      SUM(totalamount) AS totalamount,
      COUNT(*) AS num_transactions,
      AVG(totalamount) AS AOV
    FROM (
      SELECT
        id,
        chain,
        date,
        transaction_id,
        SUM(abs_purchaseamount) AS totalamount
      FROM (
        SELECT
          *,
          ABS(purchaseamount) AS abs_purchaseamount,
          CASE WHEN purchaseamount < 0 AND purchasequantity < 0 THEN 1 ELSE 0 END AS returned_flag,
          CONCAT(CAST(id AS STRING), "-", CAST(chain AS STRING), "-", CAST(date AS STRING)) AS transaction_id
        FROM
          `PROJECT.DATASET.transactions` 
          )
      WHERE returned_flag = 0
      GROUP BY
        id,
        chain,
        date,
        transaction_id )
    GROUP BY
      id ) )
GROUP BY
  AOV_binned