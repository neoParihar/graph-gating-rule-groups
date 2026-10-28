// Rule nodes for the nodes a page brings that the group's path lacks.
MATCH (g:RuleGroup {id: $group_id})
UNWIND $nodes AS n
CALL (n, g) {
  WITH n, g WHERE n.kind = 'entity'
  CREATE (r:EntityRule {id: g.id + '|' + n.key, key: n.key})
  RETURN r
  UNION
  WITH n, g WHERE n.kind = 'table'
  CREATE (r:TableRule {id: g.id + '|' + n.key, key: n.key})
  RETURN r
}
MATCH (e:Entity {key: r.key})
MERGE (r)-[:OF_KIND]->(e)
