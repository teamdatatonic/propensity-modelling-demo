-- compute AOV by total_sale_amount_[x]/total_transactions_[x]
-- this reduces 16 features down to 8
-- overwrite "features"

SELECT
  *,
  total_sale_amount_1m/total_transactions_1m AS aov_1m,
  total_sale_amount_3m/total_transactions_3m AS aov_3m,
  total_sale_amount_6m/total_transactions_6m AS aov_6m,
  total_sale_amount_12m/total_transactions_12m AS aov_12m,
  overall_sale_amount_1m/overall_transactions_1m AS overall_aov_1m,
  overall_sale_amount_3m/overall_transactions_3m AS overall_aov_3m,
  overall_sale_amount_6m/overall_transactions_6m AS overall_aov_6m,
  overall_sale_amount_12m/overall_transactions_12m AS overall_aov_12m
FROM
  `PROJECT.DATASET.features`