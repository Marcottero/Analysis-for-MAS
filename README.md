# LLM Whitelist Analysis for Multi‑Agent Systems

## Problem Description
When someone is looking to build **Multi‑Agent Systems (MAS)** need to select large language models (LLMs) that balance performance, cost, latency, and functional capabilities. Choosing the wrong model can inflate operational expenses, degrade user experience, or fail compliance requirements. This repository provides a data‑driven evaluation of publicly available LLMs to help people choose the best candidate for parser, worker, and orchestrator roles within a MAS.

## Methodology
1. **Data Ingestion** – Pull the full model catalogue from **OpenRouter** and the whitelist of approved models from a public HTML source.
2. **Metric Normalisation** – Convert raw pricing, latency, and speed figures into normalized scores.
3. **Feature Engineering** – Compute composite scores for three MAS roles:
   * **Parser / Router** – Emphasises low cost, low latency, and reasonable speed.
   * **Worker (Data Manipulator)** – Balances coding ability, cost, and speed.
   * **Orchestrator** – Prioritises intelligence, knowledge (MMLU/GPQA), and context window.
4. **Clustering & Visualisation** – K‑means clustering groups models by overall capability; heat‑maps and interactive Plotly widgets expose trade‑offs.
5. **Pareto Front Extraction** – Interactive scatter plots highlight models that dominate on selected metric pairs.

## Data Sources
- **OpenRouter API** – Full list of LLMs with pricing, context window, and architecture details. (`https://openrouter.ai/api/v1/models`)
- **ArtificialAnalysis.ai API** – Evaluation scores (Intelligence, Coding, Math, Knowledge, speed, latency). (`https://artificialanalysis.ai/api/v2/data/llms/models`)
- **Whitelist HTML** – List of models approved for my use-case. (`https://cdn.reply.com/documents/challenges/02_26/api_model_whitelist.html`)

## Repository Structure
```
project/
├─ data/                 # Raw datasets (fetched via API)
├─ notebooks/            # Notebook
├─ src/                  # Extracted logic
│   ├─ data_utils.py     # API fetchers, slug normalisation
│   └─ viz_utils.py      # Clustering, heat‑map, and interactive Plotly visualisation
├─ requirements.txt      # Fixed dependency versions
└─ README.md             # This document
```

## Acknowledgements

We thank OpenRouter (https://openrouter.ai/) and ArtificialAnalysis.ai (https://artificialanalysis.ai/) for providing the free APIs used in this analysis.