"""The rule-group memory and its confidence gate, in the order the pipeline runs:

    memory = RuleMemory(cfg); memory.build(corpus)          # src/rule_memory.py: ingest pages into rule groups,
                                                            #   consolidated by src/consolidation.py
    post = RuleGroupPosterior(memory, corpus, cfg)          # src/posterior.py: P(rule group | page evidence)
    gate = Gate(post, memory)                               # src/gate.py: confidence per rule, and the gate
    passes = GeminiPasses(cfg, corpus, gate)                # src/llm.py: call 0 and the gated pass

Cypher queries live in src/queries/, prompts in src/prompts/ (src/resources.py loads them).
"""
