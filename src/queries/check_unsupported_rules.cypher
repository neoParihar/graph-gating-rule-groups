// Build check: rule nodes no page supports (should be none).
MATCH (g:RuleGroup)-[:`_`*]->(r)
WHERE NOT EXISTS { (r)<-[:SUPPORTS]-() }
RETURN r.id AS id
