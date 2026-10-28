// A page joins a group: IN_GROUP, the running mean structure vector, and SUPPORTS from each page node to the rule of the same kind.
MATCH (g:RuleGroup {id: $group_id}), (p:Page {id: $page_id})
MERGE (p)-[m:IN_GROUP]->(g)
SET m.role = $role, m.order = $order, m.candidates = $candidates, m.passed = $passed, m.similarity = $similarity,
    g.n_pages = g.n_pages + 1,
    g.structure_sum = [i IN range(0, size(g.structure_sum) - 1) | g.structure_sum[i] + $structure[i]]
SET g.structure_vector = [x IN g.structure_sum | x / g.n_pages]
WITH g, p
MATCH (p)-[:`_`*]->(f)
MATCH (g)-[:`_`*]->(r {key: f.key})
MERGE (f)-[:SUPPORTS]->(r)
