-- for instances where both purchaseamount and purchasequantity are not < 0 (i.e not returned_flag = 1) then convert to absolute numbers
-- overwrite "cleaned" table
SELECT
  id,
  chain,
  dept,
  category,
  company,
  brand,
  date,
  productsize,
  productmeasure,
  CASE
    WHEN returned_flag = 0 THEN ABS(purchasequantity)
    ELSE purchasequantity
  END AS purchasequantity,
  CASE
    WHEN returned_flag = 0 THEN ABS(purchaseamount)
    ELSE purchaseamount
  END AS purchaseamount,
  returned_flag
FROM
  `PROJECT.DATASET.cleaned`


/*
-- test outcome
SELECT
count(*)
FROM
  `PROJECT.DATASET.transactions`
  WHERE purchaseamount < 0


SELECT
count(*)
FROM
  `PROJECT.DATASET.transactions`
  WHERE purchasequantity < 0
*/