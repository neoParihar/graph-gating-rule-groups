// The entity kinds that take part in the arithmetic check, linked to it.
MATCH (e:Entity) WHERE e.key IN $members
MATCH (i:Invariant {name: 'arithmetic_check'})
MERGE (e)-[:CONSTRAINED_BY]->(i)
