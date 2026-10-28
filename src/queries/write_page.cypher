// A page as a path of its filled entities and tables, each linked to its Entity kind.
MATCH (d:Document {doc_id: $doc_id})
MERGE (d)-[:HAS_PAGE]->(p:Page {id: $page_id})
SET p.number = $number, p.page_type = $page_type, p.signature = $signature, p.structure_vector = $structure
WITH p
UNWIND range(0, size($nodes) - 1) AS i
WITH p, i, $nodes[i] AS n
CALL (n) {
  WITH n WHERE n.kind = 'entity'
  CREATE (f:Field {id: n.id, key: n.key, bbox: n.bbox, point: n.point, label: n.label})
  RETURN f AS node
  UNION
  WITH n WHERE n.kind = 'table'
  CREATE (f:Table {id: n.id, key: n.key, region: n.bbox, point: n.point, rows: n.rows,
                   filled_columns: n.filled_columns})
  RETURN f AS node
}
WITH p, i, node ORDER BY i
WITH p, collect(node) AS chain
FOREACH (j IN range(0, size(chain) - 1) |
  FOREACH (a IN CASE WHEN j = 0 THEN [p] ELSE [chain[j - 1]] END | FOREACH (b IN [chain[j]] | MERGE (a)-[:`_`]->(b))))
WITH chain
UNWIND chain AS node
MATCH (e:Entity {key: node.key})
MERGE (node)-[:OF_KIND]->(e)
