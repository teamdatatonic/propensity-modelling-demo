-- supporting SQL queries for data quality checks and ad hoc analysis

---------- "transactions" table ----------

-- checking for NULLs
SELECT
  COUNTIF( id IS NULL) AS id,
  COUNTIF( chain IS NULL) AS  chain,
  COUNTIF( dept IS NULL) AS dept,
  COUNTIF( category IS NULL) AS category,
  COUNTIF( company IS NULL) AS company,
  COUNTIF( brand IS NULL) AS brand,
  COUNTIF( date IS NULL) AS date,
  COUNTIF( productsize IS NULL) AS productsize,
  COUNTIF( productmeasure IS NULL) AS productmeasure,
  COUNTIF( purchasequantity IS NULL) AS  purchasequantity,
  COUNTIF( purchaseamount IS NULL) AS purchaseamount
FROM
  `PROJECT.DATASET.transactions`

-- checking for duplicates
SELECT
  *,
  COUNT(*)
FROM
  `PROJECT.DATASET.transactions`
GROUP BY
  id,
  chain,
  dept,
  category,
  company,
  brand,
  date,
  productsize,
  productmeasure,
  purchasequantity,
  purchaseamount
HAVING
  COUNT(*) > 1



---------- "history" table ----------

-- checking for NULLs
SELECT
  COUNTIF( id IS NULL) AS id,
  COUNTIF( chain IS NULL) AS  chain,
  COUNTIF( offer IS NULL) AS offer,
  COUNTIF( market IS NULL) AS market ,
  COUNTIF( repeattrips IS NULL) AS repeattrips,
  COUNTIF( repeater IS NULL) AS repeater,
  COUNTIF( offerdate IS NULL) AS offerdate
FROM
  `PROJECT.DATASET.history`

-- checking for duplicates
SELECT
  *,
  COUNT(*)
FROM
  `PROJECT.DATASET.history`
GROUP BY
  id,
  chain,
  offer,
  market,
  repeattrips,
  repeater,
  offerdate
HAVING
  COUNT(*) > 1 



---------- target variable ----------
SELECT
  month,
  year,
  COUNT(*) AS num_transactions,
  COUNT(DISTINCT(brand)) AS unique_brands,
  COUNT(DISTINCT(id)) AS unique_customers
FROM (
  SELECT
    *,
    EXTRACT(MONTH
    FROM
      date) AS month,
    EXTRACT(YEAR
    FROM
      date) AS year
  FROM
    `PROJECT.DATASET.transactions` )
GROUP BY
  month,
  year
ORDER BY
  year,
  month ASC



---------- product hierarchy ----------
-- see how many categories each brand is mapped to
select
brand,
count(distinct(category)) as cnt
FROM
`PROJECT.DATASET.transactions`
GROUP BY brand
ORDER BY cnt DESC

-- see how many departments each category is mapped to
select
category,
count(distinct(dept)) as cnt
FROM
`PROJECT.DATASET.transactions`
GROUP BY category
ORDER BY cnt DESC










