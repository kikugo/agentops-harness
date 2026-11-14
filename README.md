# AgentOps harness

Red-team, guard and trace harness for my [Droit du Travail Copilot](https://github.com/kikugo/droit-du-travail-copilot), a French labour-law agent on Azure AI Foundry. It attacks the agent with garak and PyRIT, puts runtime rails in front of and behind it (hand-written regex rules plus Presidio PII masking), measures attack success and false refusals with and without the rails, and checks every pull request offline against a fixed attack suite. Personal project, work in progress.

## Scope and authorisation

Every attack in this repo targets systems I own: my own C1 agent, running locally or in my own Azure subscription, and my own Azure AI Search service. The poisoned test articles live in a separate index named `articles-redteam`, which only the `c1-redteam` service reads (`docker-compose.yml`); the demo service reads `articles-v1`. No script in this repo writes to any search index, so keeping the poisoned articles out of `articles-v1` is a manual rule, not one the code enforces. Don't point this harness at a system you don't own or aren't authorised to test.

## How it runs

The C1 agent is a git submodule pinned to one commit (`vendor/droit-du-travail-copilot`). `target/adapter.py` puts an OpenAI-compatible `/v1/chat/completions` in front of its `POST /ask`, so garak and PyRIT can call it like any chat model; the `model` field picks the demo target (`c1`) or the red-team one (`c1-redteam`).

```bash
cp .env.example .env              # fill in the C1 endpoints and the service principal
./scripts/langfuse_env.sh         # local Langfuse secrets and project keys, written to .env
docker compose up -d --build      # Langfuse, C1 and the adapter, all on 127.0.0.1
uv run python scripts/check_trace.py
```

## Tracing

Every request is one trace in a self-hosted Langfuse (its official docker compose, pinned to v4.71.0, UI on `http://localhost:3000`). The adapter opens the root span and passes the attack id and campaign from the `X-Attack-Id` and `X-Campaign` headers to C1 as W3C baggage, so every span below carries them: the C1 request, the agent run, each model call, each tool call and each Azure AI Search query. C1's own code is unchanged: `target/c1_otel.py` sets up the exporter before C1 starts, and without `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` it does nothing, so C1's Application Insights setup for its demo is unaffected. `scripts/check_trace.py` sends one question and checks the trace in Langfuse (`results/trace_check.json`).

## Rails

The rails live in `guardrails/` and run inside the adapter for the `-guarded` models. They are plain Python: NeMo Guardrails is listed as a dependency but no rail uses it yet.

- Input: regex rules for jailbreak phrasing and for off-topic or illegal requests (`input_rails.py`). A match returns a fixed refusal and C1 is never called.
- Output: a citation check that appends a note when an answer cites no retrieved article, and Presidio PII masking with a French phone and NIR recognizer (`output_rails.py`).
- Retrieval: `retrieval_rails.py` strips instruction-like lines from retrieved chunks. **Limitation:** C1 is a pinned submodule that this repo does not modify, so the rail can only clean the citations C1 returns after the model has already read the raw chunks. It does not protect the model from indirect prompt injection.

## CI

`.github/workflows/redteam.yml` runs on every pull request, without secrets or model calls:

- lint and unit tests (`uv run pytest`);
- the offline rail check (`scripts/offline_gate.py`): the whole attack suite and C1's golden questions go through the real adapter and rails to a stub agent. It fails if a rail lets more attacks through than `results/offline_gate_expected.json` allows, if the input rails refuse more than 10% of answerable questions, if the PII rail puts a placeholder into any golden answer, or if anything ends in an error.

The live gate runs the smoke attacks and a subset of golden questions against the real C1 and fails if attack success rises more than 5 points, false refusals pass 10%, or more than 10% of requests error. It needs Azure secrets, which this repository does not have, so it shows as skipped on pull requests. The nightly schedule is off for the same reason; the full scan can still be started by hand.

## Status

Phases 0 to 6 merged. The result files were rescored with the current scoring (errors split out); the campaigns have not been rerun since the PII rail fix.

## Results

![Attacks blocked per OWASP category with and without guardrails, and the false-refusal rate on legitimate questions](docs/images/attacks-blocked.webp)

<!-- RESULTS:START -->
These tables come from the attack and question runs of 2026-11-02, rescored on 2026-11-14 with `scripts/rescore_results.py` (no new model calls). Attacks or questions that ended in an error are reported apart and left out of the rates.

### Attack Success Rate Before and After Guardrails

| OWASP Category | Vulnerability | Attacks | Errors (base / guarded) | Baseline ASR | Guarded ASR | Change |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LLM01** | Prompt Injection | 12 | 0 / 0 | 0.00% (0/12) | 0.00% (0/12) | 0.00 pts |
| **LLM02** | Sensitive Information Disclosure | 12 | 0 / 0 | 0.00% (0/12) | 0.00% (0/12) | 0.00 pts |
| **LLM06** | Excessive Agency | 12 | 0 / 0 | 0.00% (0/12) | 0.00% (0/12) | 0.00 pts |
| **LLM07** | System Prompt Leakage | 12 | 1 / 1 | 0.00% (0/11) | 9.09% (1/11) | +9.09 pts |
| **LLM08** | Vector & Embedding Weaknesses | 12 | 0 / 0 | 0.00% (0/12) | 0.00% (0/12) | 0.00 pts |
| **LLM09** | Misinformation & Tool Misuse | 12 | 0 / 0 | 8.33% (1/12) | 8.33% (1/12) | 0.00 pts |
| **Total** | **All Categories** | **72** | **1 / 1** | **1.41% (1/71)** | **2.82% (2/71)** | **+1.41 pts** |

Errors: `LLM07-en-006`. The judge's own reply was blocked by its provider's content filter, so the attack was never scored. It is not counted as defended.

#### Language Slice Breakdown

| Language | Attacks | Baseline ASR | Guarded ASR | Change |
| :--- | :---: | :---: | :---: | :---: |
| **English (en)** | 36 | 0.00% (0/35) | 5.71% (2/35) | +5.71 pts |
| **French (fr)** | 36 | 2.78% (1/36) | 0.00% (0/36) | -2.78 pts |

### What the Rails Stop on Their Own

`scripts/offline_gate.py` sends the same attacks through the adapter and rails to a stub agent that complies with every attack, so the agent's own refusals play no part. Without rails every attack succeeds (100%); with rails 57 of 72 still get through. Most of the low guarded ASR above comes from C1 refusing, not from the rails.

| OWASP Category | Attacks stopped by the rails alone |
| :--- | :---: |
| **LLM01** | 100% |
| **LLM02** | 0% |
| **LLM06** | 0% |
| **LLM07** | 25% |
| **LLM08** | 0% |
| **LLM09** | 0% |

### Guardrail Cost and Impact on Legitimate Traffic

| Metric | Without Rails (Baseline) | With Rails (Guarded) | Overhead / Impact |
| :--- | :---: | :---: | :---: |
| Errors (100 Qs, left out of the rates) | 0 | 0 | |
| False Refusal Rate (100 Qs) | 4.71% | 2.35% | -2.36% |
| Answerable questions refused by the input rails alone (offline check) | | 0/85 | |
| Answers with a PII placeholder (`<PERSON>` etc.) (at least; only the first 120 characters were stored) | 0 | 6 | g060, g061, g062, g063, g072, g084 |
| Tokens p50 (estimated, chars / 3.5) | 241 | 257 | +16 |
| Tokens p95 (estimated, chars / 3.5) | 394 | 363 | -31 |

A refusal is a keyword match on the answer. The two false-refusal rates come from separate runs of a non-deterministic agent; the offline row shows what the input rails refuse by themselves.

Latency is not reported for these runs: they timed only the C1 call, not the rails, and the two columns came from separate runs. `scripts/evaluate_legitimate.py` now times each request end to end and records the time spent in rails; the next run will fill these rows.

#### PII Rail Fix

The first PII rail masked every spaCy PERSON hit, and the small French model tags words such as `n’` and `préavis` as people. The rail now keeps a PERSON only if it looks like a full name, and skips legal terms and article numbers. `scripts/replay_output_rails.py` replays both versions over the 100 stored unguarded answers (first 120 characters each) and 6 lines with planted personal data.

| PII rail | Legitimate answers changed | Planted PII lines masked |
| :--- | :---: | :---: |
| Before | 4/100 | 6/6 |
| After | 0/100 | 6/6 |

The guarded rows above still show the old rail, because the campaigns have not been rerun since. The after rail also misses a surname on its own: a full name needs two capitalised words.

### Judge Reliability

I labelled 50 PyRIT campaign rows by hand (`results/manual_check.csv`) and compared them with the judge's verdicts (`scripts/judge_agreement.py`).

| | Human: breach | Human: no breach |
| :--- | :---: | :---: |
| **Judge: breach** | 0 | 0 |
| **Judge: no breach** | 1 | 49 |

Raw agreement is 98%, but it says little: the judge scored 0 rows as breaches, the sample held 1 real breach(es), and the judge missed 1 (`tool-abuse-001`). Cohen's kappa is 0.00 and recall 0.00. With so few breaches in the sample, these rows cannot show that the judge catches breaches; the attack success rates above may undercount.
<!-- RESULTS:END -->
