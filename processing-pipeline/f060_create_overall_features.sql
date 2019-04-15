-- generation of behavioural features across all brands
-- overwrite "features" table

SELECT
  *
  EXCEPT(id_b, id_c, id_d, id_e)
FROM
  `PROJECT.DATASET.features` a
LEFT JOIN
  -- 1M
  (
  SELECT
    id AS id_b,
    COUNT(DISTINCT(transaction_id)) AS overall_transactions_1m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS overall_sale_amount_1m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS overall_sale_quantity_1m,
    SUM(returned_flag) AS overall_returned_items_1m,
    COUNT(DISTINCT(product_id)) AS overall_distinct_products_1m,
    COUNT(DISTINCT(brand)) AS overall_distinct_brands_1m,
    COUNT(DISTINCT(chain)) AS overall_distinct_chains_1m,
    COUNT(DISTINCT(category)) AS overall_distinct_category_1m,
    COUNT(DISTINCT(date)) AS overall_distinct_days_shopped_1m
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -1 MONTH)
  GROUP BY
    id_b ) b
ON
  a.customer_id = b.id_b
LEFT JOIN
  -- 3M
  (
  SELECT
    id AS id_c,
    COUNT(DISTINCT(transaction_id)) AS overall_transactions_3m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS overall_sale_amount_3m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS overall_sale_quantity_3m,
    SUM(returned_flag) AS overall_returned_items_3m,
    COUNT(DISTINCT(product_id)) AS overall_distinct_products_3m,
    COUNT(DISTINCT(brand)) AS overall_distinct_brands_3m,
    COUNT(DISTINCT(chain)) AS overall_distinct_chains_3m,
    COUNT(DISTINCT(category)) AS overall_distinct_category_3m,
    COUNT(DISTINCT(date)) AS overall_distinct_days_shopped_3m
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -3 MONTH)
  GROUP BY
    id_c ) c
ON
  a.customer_id = c.id_c
LEFT JOIN
  -- 6M
  (
  SELECT
    id AS id_d,
    COUNT(DISTINCT(transaction_id)) AS overall_transactions_6m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS overall_sale_amount_6m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS overall_sale_quantity_6m,
    SUM(returned_flag) AS overall_returned_items_6m,
    COUNT(DISTINCT(product_id)) AS overall_distinct_products_6m,
    COUNT(DISTINCT(brand)) AS overall_distinct_brands_6m,
    COUNT(DISTINCT(chain)) AS overall_distinct_chains_6m,
    COUNT(DISTINCT(category)) AS overall_distinct_category_6m,
    COUNT(DISTINCT(date)) AS overall_distinct_days_shopped_6m
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -6 MONTH)
  GROUP BY
    id_d ) d
ON
  a.customer_id = d.id_d
LEFT JOIN
  -- 12M
  (
  SELECT
    id AS id_e,
    COUNT(DISTINCT(transaction_id)) AS overall_transactions_12m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS overall_sale_amount_12m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS overall_sale_quantity_12m,
    SUM(returned_flag) AS overall_returned_items_12m,
    COUNT(DISTINCT(product_id)) AS overall_distinct_products_12m,
    COUNT(DISTINCT(brand)) AS overall_distinct_brands_12m,
    COUNT(DISTINCT(chain)) AS overall_distinct_chains_12m,
    COUNT(DISTINCT(category)) AS overall_distinct_category_12m,
    COUNT(DISTINCT(date)) AS overall_distinct_days_shopped_12m
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -12 MONTH)
  GROUP BY
    id_e ) e
ON
  a.customer_id = e.id_e