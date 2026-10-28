// Build check: groups whose path length differs from their recorded signature (should be none).
MATCH (g:RuleGroup)
WHERE g.signature <> '' AND size([(g)-[:`_`*]->(r) | r]) <> size(split(g.signature, ' > '))
RETURN g.id AS id
