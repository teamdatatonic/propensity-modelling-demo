-- impute missing values in productmeasure column (11500472 rows in total)
-- create new table based on bash script e.g. "cleaned"
SELECT
  id,
  chain,
  dept,
  category,
  company,
  brand,
  date,
  productsize,
  CASE
    WHEN productmeasure IS NULL THEN "Unknown"
    ELSE productmeasure
  END AS productmeasure,
  purchasequantity,
  purchaseamount
FROM
  `PROJECT.DATASET.transactions`