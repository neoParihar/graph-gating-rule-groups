// Where every training page went, in ingestion order ($vendor null for all vendors).
MATCH (v:Vendor)-[:HAS_DOCUMENT]->(d:Document)-[:HAS_PAGE]->(p:Page)-[m:IN_GROUP]->(g:RuleGroup)
WHERE $vendor IS NULL OR v.key = $vendor
RETURN m.order AS order, v.key AS vendor, d.doc_id AS doc_id, p.number AS page, g.id AS group,
       m.role AS role, m.candidates AS candidates, m.passed AS passed, m.similarity AS similarity,
       size([(p)-[:`_`*]->(f) | f]) AS nodes
ORDER BY order
