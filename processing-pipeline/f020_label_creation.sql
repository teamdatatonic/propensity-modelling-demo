-- label generation
-- create table "features"
SELECT
  *
  EXCEPT(id_b, brand_b, target),
  CASE
    WHEN target IS NULL THEN 0
    ELSE target
  END AS label
FROM
  `PROJECT.DATASET.customerxbrand` a
LEFT JOIN
-- target for whether customer bought in March 2013 or not
  (
  SELECT
    id AS id_b,
    brand AS brand_b,
    CASE
      WHEN COUNT(*) > 0 THEN 1
      ELSE 0
    END AS target
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date >= 'TRGT_MONTH'
    AND date < DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL 1 MONTH)
    AND returned_flag = 0
  GROUP BY
    id_b,
    brand_b ) b
ON
  a.customer_id = b.id_b
  AND a.brand = b.brand_b