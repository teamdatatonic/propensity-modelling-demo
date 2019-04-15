-- create pseudo product ID by concatenating brand, productsize and productmeasure
-- overwrite "cleaned" table
SELECT
*,
CONCAT(CAST(brand AS STRING), "-", CAST(productsize AS STRING), "-", CAST(productmeasure AS STRING)) AS product_id
FROM
`PROJECT.DATASET.cleaned`
