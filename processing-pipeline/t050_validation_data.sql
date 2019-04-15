-- create "validation" dataset
SELECT
  *
  EXCEPT(test_validation)
FROM
  `PROJECT.DATASET.test`
WHERE
  test_validation = "validation"