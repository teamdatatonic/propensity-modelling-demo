-- removing rows where purchasequantity = 0 (this only represents 0.15% of the data)
-- overwrite "cleaned" table
-- expect 349137053 rows left
SELECT
*
FROM
  `PROJECT.DATASET.cleaned`
WHERE
purchasequantity != 0