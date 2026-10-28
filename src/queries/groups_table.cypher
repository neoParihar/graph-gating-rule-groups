// One row per rule group: vendor, page type, pages, nodes on its path, and its structure.
MATCH (v:Vendor)-[:HAS_RULE_GROUP]->(g:RuleGroup)
RETURN v.key AS vendor, g.id AS group, g.document_type AS document_type, g.page AS page,
       g.n_pages AS pages, size([(g)-[:`_`*]->(r) | r]) AS nodes, g.signature AS signature
ORDER BY pages DESC, group
