  -- generation of brand-related behavioural features
  -- overwrite "features" table
SELECT
  *
  EXCEPT(id_b, brand_b, id_c, brand_c, id_d, brand_d, id_e, brand_e)
FROM
  `PROJECT.DATASET.features` a
LEFT JOIN
  -- 1M
  (
  SELECT
    id AS id_b,
    brand AS brand_b,
    COUNT(DISTINCT(transaction_id)) AS total_transactions_1m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS total_sale_amount_1m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS total_sale_quantity_1m,
    MAX(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS max_sale_amount_1m,
    MAX(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS max_sale_quantity_1m,
    SUM(returned_flag) AS total_returned_items_1m,
    COUNT(DISTINCT(product_id)) AS distinct_products_1m,
    COUNT(DISTINCT(chain)) AS distinct_chains_1m,
    COUNT(DISTINCT(category)) AS distinct_category_1m,
    COUNT(DISTINCT(date)) AS distinct_days_shopped_1m
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -1 MONTH)
  GROUP BY
    id_b,
    brand_b ) b
ON
  a.customer_id = b.id_b
  AND a.brand = b.brand_b
LEFT JOIN
  -- 3M
  (
  SELECT
    id AS id_c,
    brand AS brand_c,
    COUNT(DISTINCT(transaction_id)) AS total_transactions_3m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS total_sale_amount_3m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS total_sale_quantity_3m,
    MAX(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS max_sale_amount_3m,
    MAX(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS max_sale_quantity_3m,
    SUM(returned_flag) AS total_returned_items_3m,
    COUNT(DISTINCT(product_id)) AS distinct_products_3m,
    COUNT(DISTINCT(chain)) AS distinct_chains_3m,
    COUNT(DISTINCT(category)) AS distinct_category_3m,
    COUNT(DISTINCT(date)) AS distinct_days_shopped_3m
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -3 MONTH)
  GROUP BY
    id_c,
    brand_c ) c
ON
  a.customer_id = c.id_c
  AND a.brand = c.brand_c
LEFT JOIN
  -- 6M
  (
  SELECT
    id AS id_d,
    brand AS brand_d,
    COUNT(DISTINCT(transaction_id)) AS total_transactions_6m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS total_sale_amount_6m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS total_sale_quantity_6m,
    MAX(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS max_sale_amount_6m,
    MAX(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS max_sale_quantity_6m,
    SUM(returned_flag) AS total_returned_items_6m,
    COUNT(DISTINCT(product_id)) AS distinct_products_6m,
    COUNT(DISTINCT(chain)) AS distinct_chains_6m,
    COUNT(DISTINCT(category)) AS distinct_category_6m,
    COUNT(DISTINCT(date)) AS distinct_days_shopped_6m
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -6 MONTH)
  GROUP BY
    id_d,
    brand_d ) d
ON
  a.customer_id = d.id_d
  AND a.brand = d.brand_d
LEFT JOIN
  -- 12M
  (
  SELECT
    id AS id_e,
    brand AS brand_e,
    COUNT(DISTINCT(transaction_id)) AS total_transactions_12m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS total_sale_amount_12m,
    SUM(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS total_sale_quantity_12m,
    MAX(CASE
        WHEN returned_flag = 0 THEN purchaseamount
        ELSE 0 END) AS max_sale_amount_12m,
    MAX(CASE
        WHEN returned_flag = 0 THEN purchasequantity
        ELSE 0 END) AS max_sale_quantity_12m,
    SUM(returned_flag) AS total_returned_items_12m,
    COUNT(DISTINCT(product_id)) AS distinct_products_12m,
    COUNT(DISTINCT(chain)) AS distinct_chains_12m,
    COUNT(DISTINCT(category)) AS distinct_category_12m,
    COUNT(DISTINCT(date)) AS distinct_days_shopped_12m
  FROM
    `PROJECT.DATASET.cleaned`
  WHERE
    date < 'TRGT_MONTH'
    AND date >= DATE_ADD(DATE(CAST('TRGT_MONTH' AS TIMESTAMP)), INTERVAL -12 MONTH)
  GROUP BY
    id_e,
    brand_e ) e
ON
  a.customer_id = e.id_e
  AND a.brand = e.brand_e