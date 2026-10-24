# Responsible AI lab

Two Python sample projects implement the four requested areas: explainability, fairness, GenAI risk testing, and governance/observability. All example data is synthetic. The trailing empty numbered item in the brief is treated as a formatting artifact.

## Run locally

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
cd C:\Users\lilyq\Documents\ChatGPT\FDE\responsible-ai-lab
uv sync --cache-dir .uv-cache --extra dev --extra eval --prerelease=allow
.venv\Scripts\python -m lab.ml
.venv\Scripts\python -m lab.ai "How long does standard shipping take?"
.venv\Scripts\python -m lab.evaluate
.venv\Scripts\python -m pytest -q
```

Semantic Kernel depends on an Azure prerelease package; dependency resolution explicitly permits prereleases. Retain `uv.lock` to reproduce the tested environment. The `eval` extra installs external judges; omit it for the basic demos, and exclude the integration-import test with `pytest -q -k 'not eval_integrations_import'`.

## Sample ML project: training-program selection audit

`lab/ml.py` generates 2,400 applicants with skills, experience, and a feature correlated with a synthetic group. Historical selection labels deliberately favor one group. Group membership is excluded from classifier inputs, but the proxy still permits disparities.

1. Split into 60% training, 20% calibration, and 20% test data, stratified by group and label.
2. Train a logistic-regression baseline.
3. Use AI Fairness 360 Reweighing on training data and fit a second classifier with its sample weights.
4. Fit Fairlearn ThresholdOptimizer on calibration data, targeting demographic parity.
5. Compare all three on the untouched test set using accuracy, selection rates, TPR/FPR, demographic parity, equalized odds, and AIF360 disparate impact.
6. Explain the **baseline** using SHAP and LIME. Save numerical SHAP contributions and an interactive LIME HTML explanation.

Outputs: `artifacts/ml_report.json`, `artifacts/lime_explanation.html`.

SHAP contributions sum with the base value to the model's log odds. LIME fits a local surrogate in probability space and reports its fit score. These describe model behavior, not causation or a model's private reasoning. The explanations do not describe the separately mitigated models. See [SHAP LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html) and [LIME](https://github.com/marcotcr/lime).

Fairness metrics can conflict; equal selection rates do not establish fairness. Historical labels themselves are biased, so agreement with them is not an ethical quality measure. This small experiment has no uncertainty intervals or intersectional audit. ThresholdOptimizer requires group membership at inference, and randomizes some decisions. Reweighing need not improve every metric. See [Fairlearn postprocessing](https://fairlearn.org/v0.13/user_guide/mitigation/postprocessing.html) and [AIF360 Reweighing](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.preprocessing.Reweighing.html).

## Sample AI project: governed support RAG

`lab/ai.py` connects a LangChain runnable pipeline to a Semantic Kernel read-only knowledge-search plugin. It retrieves from three support documents and returns an answer plus retrieved source IDs. In default mode it extracts document text without calling an LLM. `--live` replaces extraction with a chat model. Source IDs indicate retrieval provenance; they are not proof that every generated claim is supported.

Controls implemented:

- Input length limits and normalized pattern screening for selected injection attempts.
- Screening of retrieved documents before they enter generation.
- A Semantic Kernel invocation filter allowing only `knowledge.search`.
- System instructions separated from JSON-encoded user/context data.
- No shell, network-action, file-write, or model-selected tools.
- Output screening, empty-retrieval abstention, bounded model timeouts/retries.
- JSONL audit events with trace ID, latency, status, mode, policy version, and corpus hash; prompts/responses are excluded.

Prompt screening and instruction separation are defense layers, not complete injection prevention. Paraphrases, obfuscation, multilingual attacks and novel payloads can evade these patterns. The tool allowlist provides a stronger capability boundary. Semantic Kernel is an orchestration framework, not an automatic security product; see [Microsoft's prompt-injection guidance](https://learn.microsoft.com/en-us/semantic-kernel/concepts/prompts/prompt-injection-attacks). A production service would also need authentication, authorization, tenant isolation, rate limits, durable audit storage and adversarial testing.

## GenAI risk evaluation

`lab/evaluate.py` runs normal grounding, unsupported-question abstention, direct injection, secret disclosure, poisoned retrieval and Unicode-obfuscation cases. Offline exact-match checks are deterministic regression tests, not LLM hallucination measurements. The command returns a nonzero exit code on failure.

For live generation and judge scoring, configure these variables in your own shell:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:GENERATOR_MODEL = "your-generation-model-id"
$env:JUDGE_MODEL = "your-judge-model-id"
.venv\Scripts\python -m lab.ai --live "How long does standard shipping take?"
.venv\Scripts\python -m lab.evaluate --live --judges
```

Live calls send synthetic evaluation inputs to the provider and incur its usage charges. Secrets are read from environment variables, never committed. `--judges` without `--live` evaluates the extractive baseline. Keep the default judge status `not_run` distinct from passed.

| Framework | Implementation |
|---|---|
| DeepEval | FaithfulnessMetric for retrieved evidence; GEval for expected answers, refusals and abstention |
| RAGAS | Faithfulness over response and retrieved contexts |
| TruLens | OpenAI provider's groundedness score and rationale |

No-context refusal cases skip groundedness as inapplicable. All applicable judge scores must be finite and at least 0.8. Judge errors propagate; no fabricated fallback scores are used. Exact-match expectations are deliberately strict and may fail valid live paraphrases; review the case report rather than automatically loosening a gate. Judges are fallible, and the tiny test set cannot certify alignment or security. Use independently labeled, representative cases before deployment.

Evaluation references: [DeepEval faithfulness](https://deepeval.com/docs/metrics-faithfulness), [RAGAS metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/), [TruLens](https://www.trulens.org/). RAGAS and LangChain use the 0.3 compatibility APIs deliberately, because these evaluator versions import modules removed in LangChain 1.x; dependencies are bounded accordingly. Optional DeepEval/RAGAS telemetry is disabled by default, and the unused DeepEval pytest plugin is disabled to avoid import-time telemetry initialization.

## Observed sample results

Validation: 11 tests passed, including all evaluator imports, tool authorization, audit privacy, injection cases, fairness cross-checks and SHAP additivity. Both offline command-line demos completed. Live generation and external judge requests were not run; they require user-configured credentials and model IDs. SHAP emitted three upstream plotting deprecation warnings.

The seed-42 held-out run produced:

| Model | Accuracy | Demographic parity gap | Equalized odds gap |
|---|---:|---:|---:|
| Baseline | 0.719 | 0.088 | 0.054 |
| AIF360 reweighed | 0.719 | 0.004 | 0.214 |
| Fairlearn threshold | 0.708 | 0.025 | 0.133 |

Both mitigations reduce the selection-rate disparity here, while increasing equalized-odds disparity. This is a concrete tradeoff to review, not evidence that either model is universally fairer.

## Files

| File | Purpose |
|---|---|
| `lab/ml.py` | Model training, explanations, bias detection and mitigation |
| `lab/ai.py` | Retrieval, generation, governance, observability |
| `lab/evaluate.py` | Risk cases, external judges, pass/fail gates |
| `tests/test_lab.py` | Regression tests and integration import checks |
| `artifacts/` | Generated reports and explanations |

The examples are educational; they do not make real selection decisions or assert production readiness.
