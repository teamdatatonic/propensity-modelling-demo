-- join on features for downsampled data
-- overwrite "downsampled"
SELECT
  b.*
FROM (
  SELECT
    customer_id AS customer_id_a,
    brand AS brand_a,
    label AS label_a
  FROM
    `PROJECT.DATASET.downsampled` ) a
INNER JOIN (
  SELECT
    *
  FROM
    `PROJECT.DATASET.train` ) b
ON
  a.customer_id_a = b.customer_id
  AND a.brand_a = b.brand