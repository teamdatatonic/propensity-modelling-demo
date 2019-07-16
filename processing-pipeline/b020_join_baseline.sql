-- overwrite baseline table
SELECT
  customer_id,
  brand,
  label,
  predicted_label
FROM (
  SELECT
    customer_id,
    brand,
    label
  FROM
    `PROJECT.DATASET.test` ) a
INNER JOIN (
  SELECT
    CAST(customer_id AS STRING) AS customer_id_b,
    CAST(brand AS STRING) AS brand_b,
    label AS predicted_label
  FROM
    `PROJECT.DATASET.baseline` ) b
ON
  a.customer_id = b.customer_id_b
  AND a.brand = b.brand_b