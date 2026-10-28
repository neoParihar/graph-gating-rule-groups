// Every node of one group's path, in path order: entities with their rule, tables with their usual columns.
MATCH (v:Vendor)-[:HAS_RULE_GROUP]->(g:RuleGroup {id: $id})
MATCH path = (g)-[:`_`*]->(r)
RETURN v.key AS vendor, g.id AS group, g.n_pages AS group_pages,
       CASE WHEN r:TableRule THEN 'table' ELSE 'entity' END AS kind, r.key AS name, length(path) AS order,
       r.fill_rate AS fill_rate, r.support AS support, r.n_boxes AS n_boxes, r.strength AS strength,
       coalesce(r.bbox_median, r.region_median) AS bbox, r.reliability AS reliability,
       CASE WHEN r:TableRule THEN r.columns ELSE r.top_labels END AS top_labels
ORDER BY order
