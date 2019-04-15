-- extract top 1000 brands by number of purchases
-- save as "top_brands" table
SELECT
  brand
FROM (
  SELECT
    brand,
    COUNT(*) AS num_purchases
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
  GROUP BY
    brand
  ORDER BY
    num_purchases DESC
  LIMIT
    1000 )
