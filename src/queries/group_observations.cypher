// Every observation supporting every rule of one group.
MATCH (g:RuleGroup {id: $group_id})-[:`_`*]->(r)<-[:SUPPORTS]-(f)<-[:`_`*]-(p:Page)-[m:IN_GROUP]->(g)
MATCH (d:Document)-[:HAS_PAGE]->(p)
RETURN r.id AS rule, labels(r)[0] AS kind, r.key AS key, m.order AS order, d.doc_id AS doc_id,
       f.point AS point, coalesce(f.bbox, f.region) AS bbox, coalesce(f.label, '') AS label,
       f.rows AS rows, f.filled_columns AS filled_columns
ORDER BY rule, order
