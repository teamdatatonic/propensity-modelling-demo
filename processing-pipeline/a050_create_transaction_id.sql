-- creating a pseudo transaction ID by concatenating id, chain and date
-- overwrite "cleaned" table
SELECT
  CONCAT(CAST(id AS STRING), "-", CAST(chain AS STRING), "-", CAST(date AS STRING)) AS transaction_id,
  *
FROM
  `PROJECT.DATASET.cleaned`