// The vendor of a page, created when its key is new; the printed name is added to its names.
MERGE (v:Vendor {key: $key})
ON CREATE SET v.names = []
SET v.names = CASE WHEN $name IN v.names THEN v.names ELSE v.names + $name END
