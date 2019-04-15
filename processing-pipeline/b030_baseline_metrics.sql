-- save as baseline_metrics
SELECT
  *,
  2*((precision*recall)/(precision+recall)) as f1,
  (1 + recall)/2 AS auc
FROM (
  SELECT
    *,
    (true_positive + true_negative) / total_points AS accuracy,
    true_positive / (true_positive + false_negative) AS recall,
    true_positive / (true_positive + false_positive) AS precision,
    false_positive / (false_positive + true_negative) AS fpr
  FROM (
    SELECT
      SUM(CASE
          WHEN predicted_label = 0 AND label = 0 THEN 1
          ELSE 0 END) AS true_negative,
      SUM(CASE
          WHEN predicted_label = 1 AND label = 1 THEN 1
          ELSE 0 END) AS true_positive,
      SUM(CASE
          WHEN predicted_label = 0 AND label = 1 THEN 1
          ELSE 0 END) AS false_negative,
      SUM(CASE
          WHEN predicted_label = 1 AND label = 0 THEN 1
          ELSE 0 END) AS false_positive,
      SUM(CASE
          WHEN label = 0 THEN 1
          ELSE 0 END) AS zeros,
      SUM(CASE
          WHEN label = 1 THEN 1
          ELSE 0 END) AS ones,
      COUNT(*) AS total_points
    FROM
      `PROJECT.DATASET.baseline` ) )