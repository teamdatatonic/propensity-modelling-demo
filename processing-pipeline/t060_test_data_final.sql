-- overwrite "test" to be left with the final test dataset
SELECT
  *
  EXCEPT(test_dev)
FROM
  `PROJECT.DATASET.test`
WHERE
  test_dev = "test"