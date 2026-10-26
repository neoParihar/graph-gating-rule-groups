"""Readable building blocks of the confidence-gating demo (ported from gating.ipynb, behaviour unchanged).

    graph = MemoryGraph(cfg); graph.build(corpus)          # write and consolidate memory
    post = LayoutPosterior(graph, corpus, cfg)             # P(layout | evidence)
    gate = Gate(post, graph)                               # confidence per rule, and the gate decision
"""
