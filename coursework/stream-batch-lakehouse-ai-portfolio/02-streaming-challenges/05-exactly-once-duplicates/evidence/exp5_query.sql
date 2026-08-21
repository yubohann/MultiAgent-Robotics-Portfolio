USE paimon.c05;

SELECT 'Append Table' AS t,
       COUNT(*) AS total_records,
       COUNT(DISTINCT order_id) AS unique_orders,
       COUNT(*) - COUNT(DISTINCT order_id) AS dup_count
FROM append_table_demo000000
UNION ALL
SELECT 'PK Table',
       COUNT(*),
       COUNT(DISTINCT order_id),
       COUNT(*) - COUNT(DISTINCT order_id)
FROM pk_table_demo000000;

SELECT order_id, COUNT(*) AS dup_count
FROM append_table_demo000000
GROUP BY order_id
HAVING COUNT(*) > 1
ORDER BY dup_count DESC, order_id
LIMIT 10;

SELECT order_id, COUNT(*) AS dup_count
FROM pk_table_demo000000
GROUP BY order_id
HAVING COUNT(*) > 1
LIMIT 10;
