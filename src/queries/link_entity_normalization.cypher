// Each entity kind to the normalization rule of its value type ($types: {entity key: value type}).
MATCH (e:Entity) WHERE e.key IN keys($types)
MATCH (r:NormalizationRule {name: $types[e.key]})
MERGE (e)-[:NORMALIZED_BY]->(r)
