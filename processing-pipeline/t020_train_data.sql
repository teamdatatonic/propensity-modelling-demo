-- train table
-- create "train"

SELECT
  *
  EXCEPT(train_test)
FROM
  `PROJECT.DATASET.features`
WHERE
  train_test = "train"