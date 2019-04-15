-- extracting individual unit price by dividing the total purchase amount by the quantity
-- overwrite "cleaned" table
SELECT
  *,
  purchaseamount/purchasequantity AS productprice
FROM
  `PROJECT.DATASET.cleaned`