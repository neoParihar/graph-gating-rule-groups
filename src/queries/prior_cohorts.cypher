// Box centres of each entity over pages of one (vendor, page type, structure) cohort, for the reliability prior.
// Cohorts come from the document paths, not the groups, so pages the range check split apart still count together.
MATCH (v:Vendor)-[:HAS_DOCUMENT]->(:Document)-[:HAS_PAGE]->(p:Page)-[:`_`*]->(f:Field)
WHERE f.point IS NOT NULL
RETURN v.key AS vendor, p.page_type AS page, p.signature AS signature, f.key AS key, collect(f.point) AS points
