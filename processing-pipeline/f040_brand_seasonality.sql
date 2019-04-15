-- brand seasonality feature (based on how many months they have bought in last 12 months)
-- overwrite "features"
SELECT
  *
  EXCEPT(id_b, brand_b, brand_seasonality),
  CASE
    WHEN brand_seasonality IS NULL THEN 0
    ELSE brand_seasonality
  END AS brand_seasonality
FROM
  `PROJECT.DATASET.features` a
LEFT JOIN
  -- seasonality
  (
  SELECT
    id AS id_b,
    brand AS brand_b,
    COUNT(DISTINCT(EXTRACT(MONTH FROM date))) AS brand_seasonality
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -12 MONTH)
  GROUP BY
    id_b,
    brand_b ) b
ON
  a.customer_id = b.id_b
  AND a.brand = b.brand_b