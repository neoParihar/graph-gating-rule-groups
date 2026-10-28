// Remove the unnamed edges of a group's path, before it is re-chained.
MATCH (:RuleGroup {id: $group_id})-[:`_`*0..]->()-[e:`_`]->() DELETE e
