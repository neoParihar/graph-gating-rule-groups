// One rule node's consolidated properties, by id ("<group id>|<entity key>").
MATCH (r {id: $id}) WHERE r:EntityRule OR r:TableRule
RETURN r.bbox_center AS center, coalesce(r.bbox_median, r.region_median) AS median, r.range AS range,
       r.support AS support, r.n_boxes AS n_boxes, r.loo_hits AS loo_hits, r.reliability AS reliability,
       r.strength AS strength, r.fill_rate AS fill_rate, r.top_labels AS top_labels
