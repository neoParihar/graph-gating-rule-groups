// Per page type: pages, and how often each entity is present with its labels.
MATCH (p:Page)
WITH p.page_type AS page, count(p) AS pages
OPTIONAL MATCH (:Page {page_type: page})-[:`_`*]->(f)
RETURN page, pages, f.key AS key, count(f) AS n_filled, collect(CASE WHEN f.label <> '' THEN f.label END) AS labels
