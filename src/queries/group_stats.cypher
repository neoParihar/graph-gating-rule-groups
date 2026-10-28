// Every rule group with its rule nodes, for the posterior.
MATCH (v:Vendor)-[:HAS_RULE_GROUP]->(g:RuleGroup)
OPTIONAL MATCH (g)-[:`_`*]->(r)
RETURN v.key AS vendor, g.id AS group, g.page AS page, g.n_pages AS n_pages, labels(r)[0] AS kind, r.key AS key,
       size([(r)<-[:SUPPORTS]-() | 1]) AS n_supported, r.range AS range, r.reliability AS reliability,
       r.bbox_median AS bbox_median, r.strength AS strength, r.top_labels AS top_labels
