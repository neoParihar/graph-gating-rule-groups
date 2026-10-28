// For each candidate group: how many of the page's nodes the group shares, and how many of those lie inside the group's range.
// A node without a usable box, or a rule without a range yet, cannot contradict the group.
UNWIND $groups AS gid
MATCH (g:RuleGroup {id: gid})
UNWIND $nodes AS n
MATCH (g)-[:`_`*]->(r {key: n.key})
WITH g, n, n.point IS NULL OR r.range IS NULL OR
     (r.range[0] <= n.point[0] <= r.range[2] AND r.range[1] <= n.point[1] <= r.range[3]) AS inside
RETURN g.id AS group, count(n) AS checked, sum(CASE WHEN inside THEN 1 ELSE 0 END) AS inside,
       collect(CASE WHEN inside THEN null ELSE n.key END) AS outside
