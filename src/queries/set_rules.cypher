// Write consolidated properties onto rule nodes.
UNWIND $rules AS r
CALL (r) {
  MATCH (x:EntityRule {id: r.id}) SET x += r.props
  UNION
  MATCH (x:TableRule {id: r.id}) SET x += r.props
}
