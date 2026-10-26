# AgentOps harness

Red-team, guard and trace harness for my [Droit du Travail Copilot](https://github.com/kikugo/droit-du-travail-copilot), a French labour-law agent on Azure AI Foundry. It attacks the agent with garak and PyRIT, adds NeMo Guardrails, measures attack success, false refusals and the latency cost of the rails, and gates pull requests on a fixed attack suite. Personal project, work in progress.

## Scope and authorisation

Every attack in this repo targets systems I own: my own C1 agent, running locally or in my own Azure subscription, and my own Azure AI Search service. The poisoned test articles live in a separate index named `articles-redteam`; the demo index never contains them. Don't point this harness at a system you don't own or aren't authorised to test.

## Status

Phase 0 (target and test index) in progress. Results tables will be generated from `results/*.json`.
