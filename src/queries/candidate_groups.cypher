// The vendor's rule groups for one page type, with their structure vectors (the cosine is computed in Python).
MATCH (:Vendor {key: $vendor})-[:HAS_RULE_GROUP]->(g:RuleGroup {document_type: $doc_type, page: $page_type})
RETURN g.id AS group, g.n_pages AS n_pages, g.structure_vector AS structure_vector ORDER BY g.id
