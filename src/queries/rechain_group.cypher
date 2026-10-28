// Chain a group's rule nodes in schema order and record its signature.
MATCH (g:RuleGroup {id: $group_id})
SET g.signature = $signature
WITH g
UNWIND range(0, size($ids) - 1) AS j
CALL (g, j) {
  MATCH (b {id: $ids[j]}) WHERE b:EntityRule OR b:TableRule
  OPTIONAL MATCH (prev {id: $ids[j - 1]}) WHERE j > 0 AND (prev:EntityRule OR prev:TableRule)
  WITH b, CASE WHEN j = 0 THEN g ELSE prev END AS a
  MERGE (a)-[:`_`]->(b)
}
