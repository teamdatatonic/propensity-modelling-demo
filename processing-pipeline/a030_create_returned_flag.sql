-- creating a flag for returned items
-- overwrite "cleaned" table
SELECT
  *,
  CASE
    WHEN purchasequantity < 0 AND purchaseamount < 0 THEN 1
    ELSE 0
  END AS returned_flag
FROM
  `PROJECT.DATASET.cleaned`