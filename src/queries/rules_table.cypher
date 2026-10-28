// One row per entity rule with a range: its consolidated statistics.
MATCH (v:Vendor)-[:HAS_RULE_GROUP]->(g:RuleGroup)-[:`_`*]->(r:EntityRule)
WHERE r.range IS NOT NULL
RETURN v.key AS vendor, g.id AS group, g.n_pages AS pages, r.key AS entity, r.fill_rate AS fill_rate,
       r.n_boxes AS n_boxes, r.support AS support, r.spread AS spread, r.loo_hits AS loo_hits,
       r.reliability AS reliability, r.strength AS strength, r.range AS range, r.top_labels AS top_labels
