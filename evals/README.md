# Evals

Fixed task sets for scoring answer quality. Every answer is graded by code against a known answer, so no model ever grades another model or itself.

| File | Purpose |
|---|---|
| [`core.yaml`](core.yaml) | 24 original tasks, six per category, from easy to hard: exact match, JSON generation, coding against hidden tests, and multi-step reasoning |

Each task has a `reference` (a correct answer its grader must accept) and a `wrong` answer (a plausible mistake it must reject). The tests and every run check both, so a broken grader can't produce a score. Changing a file changes its hash, and scores only compare within one hash.

An experiment uses a suite through its `quality:` section; see [docs/reference.md](../docs/reference.md#quality-scoring).
