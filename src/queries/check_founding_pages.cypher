// Build check: per group, how many pages founded it (should be exactly one) and its page count.
MATCH (g:RuleGroup)
OPTIONAL MATCH (g)<-[m:IN_GROUP {role: 'founded'}]-()
RETURN g.id AS id, count(m) AS founded, g.n_pages AS n
