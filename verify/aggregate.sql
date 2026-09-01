-- Recompute reports/counterfactual_summary.csv and reports/edge_reading_control.csv
-- from the per-narration records in reports/counterfactual.json.
--
-- Both published tables come out of one pandas groupby in
-- experiments/counterfactual.py. Nothing downstream would notice if that
-- aggregation were wrong, because every figure and every README table reads its
-- output. This derives the same rows with nothing but SQL, Wilson interval
-- included, and verify/verify.sh diffs the two byte for byte.
--
-- Run: sqlite3 -init verify/aggregate.sql :memory: ""

CREATE TEMP TABLE rec AS
SELECT json_extract(value, '$.model')                  AS model,
       json_extract(value, '$.kind')                   AS kind,
       json_extract(value, '$.explainer')              AS explainer,
       json_extract(value, '$.subgraph')               AS subgraph,
       json_extract(value, '$.label')                  AS label,
       json_extract(value, '$.agrees_with_structure')  AS agrees_structure,
       json_extract(value, '$.agrees_with_label')      AS agrees_label,
       json_extract(value, '$.cited_valid')            AS cited_valid,
       json_extract(value, '$.n_cited')                AS n_cited
FROM json_each(readfile('reports/counterfactual.json'));

-- Every published proportion as one long table: which cell, which metric, how
-- many successes out of how many trials. Citation validity pools node ids
-- rather than narrations, which is why its denominator is a different column.
CREATE TEMP VIEW stat AS
    SELECT model, explainer, subgraph, label, 'structure' AS metric,
           SUM(agrees_structure) AS k, COUNT(*) AS n
    FROM rec WHERE kind = 'narrate' GROUP BY 1, 2, 3, 4
    UNION ALL
    SELECT model, explainer, subgraph, label, 'label',
           SUM(agrees_label), COUNT(*)
    FROM rec WHERE kind = 'narrate' GROUP BY 1, 2, 3, 4
    UNION ALL
    SELECT model, explainer, subgraph, label, 'citation',
           SUM(cited_valid), SUM(n_cited)
    FROM rec WHERE kind = 'narrate' GROUP BY 1, 2, 3, 4
    UNION ALL
    SELECT model, explainer, '', '', 'control',
           SUM(agrees_structure), COUNT(*)
    FROM rec WHERE kind = 'control' GROUP BY 1, 2;

-- The Wilson score interval, written out once. Not the normal approximation:
-- four of these cells sit at exactly 0.000 or 1.000, where that interval runs
-- outside [0, 1]. z = 1.96, so z^2 = 3.8416 and z^2 / 2 = 1.9208.
-- Formatted to three decimals in pandas' layout, so the comparison is against
-- the published string rather than a re-rounded number.
CREATE TEMP VIEW ci AS
SELECT model, explainer, subgraph, label, metric, n,
       printf('%.3f [%.3f,%.3f]', p,
              max(0.0, (p + 1.9208 / n - 1.96 * half) / (1 + 3.8416 / n)),
              min(1.0, (p + 1.9208 / n + 1.96 * half) / (1 + 3.8416 / n))) AS s
FROM (SELECT *, 1.0 * k / n AS p,
             sqrt(1.0 * k / n * (1 - 1.0 * k / n) / n + 0.9604 / (n * n)) AS half
      FROM stat);

.mode csv
.headers on

SELECT a.model, a.explainer, a.subgraph, a.label, a.n,
       a.s AS structure_agreement,
       b.s AS label_agreement,
       c.s AS citation_validity
FROM ci a
JOIN ci b USING (model, explainer, subgraph, label)
JOIN ci c USING (model, explainer, subgraph, label)
WHERE a.metric = 'structure' AND b.metric = 'label' AND c.metric = 'citation'
ORDER BY 1, 2, 3, 4;

.headers off
.print ---
.headers on

SELECT model, explainer, n, s AS edge_reading_accuracy
FROM ci WHERE metric = 'control' ORDER BY 1, 2;
