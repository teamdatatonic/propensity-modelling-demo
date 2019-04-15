-- train / test label split into 80 : 20
-- overwrite "features" table

SELECT
  *,
  CASE
    WHEN RAND() < 0.8 THEN "train"
    ELSE "test"
  END AS train_test
FROM
  `PROJECT.DATASET.features`