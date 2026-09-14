"""Text-to-SQL evaluation harness over the Spider dev set.

Modules:
    paths    - repo-relative locations of data, cases, runs
    sqlexec  - sandboxed, read-only SQL execution against a Spider database
    grader   - execution-match grading of a predicted query against gold
    testset  - loading and versioning of cases.yaml
    prompt   - prompt rendering (schema + question), versioned
    models   - model providers (Anthropic, plus oracle / null baselines)
    runner   - runs a model over the test set, writes a run directory
    report   - metrics, confidence intervals, run comparison
    audit    - human-adjudication sheet for a run's failures
"""
