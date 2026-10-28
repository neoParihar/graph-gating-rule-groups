// A new, empty rule group under its vendor.
MATCH (v:Vendor {key: $vendor})
CREATE (v)-[:HAS_RULE_GROUP]->(:RuleGroup {id: $group_id, document_type: $doc_type, page: $page_type,
                                           n_pages: 0, structure_sum: $zeros})
