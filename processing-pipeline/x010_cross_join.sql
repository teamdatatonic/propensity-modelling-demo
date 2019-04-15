-- cross join customers (with at least 1 transaction with top brands) and top 1000 brands that were active prior to March 2013
-- create table "customerxbrand"

SELECT
  *
FROM (
  SELECT
    DISTINCT(id) AS customer_id
  FROM (
    SELECT
      id,
      brand
    FROM
      `PROJECT.DATASET.cleaned`
    WHERE
      date < 'TRGT_MONTH'
    GROUP BY
      id,
      brand ) a
  INNER JOIN (
    SELECT
      brand AS brand_b
    FROM
      `PROJECT.DATASET.top_brands` ) b
  ON
    a.brand = b.brand_b )
CROSS JOIN (
  SELECT
    *
  FROM
    `PROJECT.DATASET.top_brands` )