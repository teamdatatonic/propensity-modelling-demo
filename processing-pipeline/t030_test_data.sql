-- test table
-- create "test"

SELECT
  *
  EXCEPT(train_test)
FROM
  `PROJECT.DATASET.features`
WHERE
  train_test = "test"