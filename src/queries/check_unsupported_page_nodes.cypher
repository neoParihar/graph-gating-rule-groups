// Build check: page nodes that support no rule of their own group (should be none).
MATCH (p:Page)-[:IN_GROUP]->(g:RuleGroup), (p)-[:`_`*]->(f)
WHERE NOT EXISTS { (f)-[:SUPPORTS]->()<-[:`_`*]-(g) }
RETURN f.id AS id
