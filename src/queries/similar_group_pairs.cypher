// Rule groups of different vendors on the same page type whose structure vectors (which nodes, and roughly where)
// are similar. Neo4j's cosine is rescaled to 0..1, as (1 + cos) / 2.
MATCH (va:Vendor)-[:HAS_RULE_GROUP]->(a:RuleGroup), (vb:Vendor)-[:HAS_RULE_GROUP]->(b:RuleGroup)
WHERE va.key < vb.key AND a.page = b.page
WITH va, vb, a, b, vector.similarity.cosine(a.structure_vector, b.structure_vector) AS sim
WHERE sim >= $min
RETURN va.key AS vendor_a, a.id AS group_a, a.n_pages AS pages_a,
       vb.key AS vendor_b, b.id AS group_b, b.n_pages AS pages_b, a.page AS page, sim
ORDER BY sim DESC
