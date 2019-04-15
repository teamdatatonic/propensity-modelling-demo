-- promotional sensitive feature
-- overwrite "features" table

SELECT
  * EXCEPT(id, repeater)
FROM (
  SELECT
    *,
    CASE
      WHEN repeater = "t" THEN 1
      ELSE 0
    END AS promo_sensitive
  FROM
    `PROJECT.DATASET.features` a
  LEFT JOIN (
    SELECT
      repeater,
      id
    FROM
      `PROJECT.DATASET.history` ) b
  ON
    a.customer_id = b.id )