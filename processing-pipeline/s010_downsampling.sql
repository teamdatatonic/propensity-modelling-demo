-- downsampling due to class imbalance
-- 17,813,514 rows now in total
-- legacy SQL = true
-- save as "downsampled"

-- TODO: best practice is to use a hashing function e.g. FARM_FINGERPRINT

SELECT
  customer_id,
  brand,
  label
FROM
  -- taking all rows (8511369) where label = 1
  (
  SELECT
    CAST(customer_id AS STRING) AS customer_id,
    CAST(brand AS STRING) AS brand,
    label
  FROM
    [PROJECT:DATASET.train]
  WHERE
    label = 1 ),
  -- randomly taking ~1/2 rows where label is zero
  (
  SELECT
    CAST(customer_id AS STRING) AS customer_id,
    CAST(brand AS STRING) AS brand,
    label
  FROM
    [PROJECT:DATASET.train]
  WHERE
    label = 0
    AND RAND(42) < 0.02 ),
  -- taking ~1/2 rows where we know we have some interactions in the 12M prior
  -- and label = 0
  (
  SELECT
    CAST(a.id AS STRING) AS customer_id,
    CAST(a.brand AS STRING) AS brand,
    b.label AS label
  FROM (
    SELECT
      CAST(id AS STRING) AS id,
      CAST(brand AS STRING) AS brand
    FROM
      [PROJECT:DATASET.cleaned]
    WHERE
      date < CAST('2013-03-01' AS DATE)
    GROUP BY
      id,
      brand ) a
  INNER JOIN (
    SELECT
      CAST(customer_id AS STRING) AS id_b,
      CAST(brand AS STRING) AS brand_b,
      label
    FROM
      [PROJECT:DATASET.train] ) b
  ON
    a.id = b.id_b
    AND a.brand = b.brand_b
  WHERE
    label = 0
    AND RAND(42) < 0.15 )
GROUP BY customer_id, brand, label