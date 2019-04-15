-- overwrite "test"
-- to split the test set into test and validation set
SELECT
  *,
  CASE
    WHEN RAND() < 0.5 THEN "test"
    ELSE "validation"
  END AS test_validation
FROM
  [PROJECT:DATASET.test]