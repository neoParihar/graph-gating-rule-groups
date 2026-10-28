// A training document under its vendor.
MATCH (v:Vendor {key: $vendor})
MERGE (v)-[:HAS_DOCUMENT]->(d:Document {doc_id: $doc_id})
SET d.document_type = $doc_type
