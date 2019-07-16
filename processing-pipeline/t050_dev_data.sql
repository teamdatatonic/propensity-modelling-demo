-- create "dev" table
SELECT
  *
  EXCEPT(test_dev)
FROM
  `PROJECT.DATASET.test`
WHERE
  test_dev = "dev"