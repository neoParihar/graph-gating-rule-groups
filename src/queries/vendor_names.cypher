// The printed names of every known vendor (not the unidentified vendor).
MATCH (v:Vendor) WHERE v.key <> $unidentified
RETURN v.key AS vendor, v.names AS names
