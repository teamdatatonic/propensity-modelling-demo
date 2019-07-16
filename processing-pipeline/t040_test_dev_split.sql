-- overwrite "test"
-- to split the table into test and dev sets

-- TODO: best practice is to use a hashing function e.g. FARM_FINGERPRINT

SELECT
  *,
  CASE
    WHEN RAND() < 0.5 THEN "test"
    ELSE "dev"
  END AS test_dev
FROM
  [PROJECT:DATASET.test]