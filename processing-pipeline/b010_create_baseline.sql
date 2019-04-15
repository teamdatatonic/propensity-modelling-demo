-- create baseline where label is if customer bought in February 2013, they will also buy in March 2013
-- save as "baseline"

SELECT
  * 
  EXCEPT(id_b, brand_b, target),
  CASE
    WHEN target IS NULL THEN 0
    ELSE target
  END AS label
FROM (
  SELECT
    customer_id,
    brand
  FROM
    `PROJECT.DATASET.validation`) a
LEFT JOIN
  -- target for whether customer in test set bought in February 2013 or not
  (
  SELECT
    CAST(id AS STRING) AS id_b,
    CAST(brand AS STRING) AS brand_b,
    CASE
      WHEN COUNT(*) > 0 THEN 1
      ELSE 0
    END AS target
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date >= '2013-02-01'
    AND date < DATE_ADD(DATE(CAST('2013-02-01' AS TIMESTAMP)), INTERVAL 1 MONTH)
    AND returned_flag = 0
  GROUP BY
    id_b,
    brand_b ) b
ON
  a.customer_id = b.id_b
  AND a.brand = b.brand_b