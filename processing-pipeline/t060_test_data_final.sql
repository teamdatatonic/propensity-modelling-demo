-- overwrite "test" to be left with the final test dataset
SELECT
  *
  EXCEPT(test_validation)
FROM
  `PROJECT.DATASET.test`
WHERE
  test_validation = "test"