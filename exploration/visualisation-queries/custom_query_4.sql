-- custom SQL query for Tableau dashboard
-- visualisation: product mapping
SELECT
dept,
COUNT(DISTINCT(category)) AS distinct_category,
COUNT(DISTINCT(brand)) AS distinct_brands
FROM
`PROJECT.DATASET.transactions`
GROUP BY dept
ORDER BY distinct_brands