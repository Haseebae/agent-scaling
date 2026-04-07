# Reproduction Guide

This document provides step-by-step instructions for reproducing all experiments in the paper.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [API Keys Required](#api-keys-required)
3. [Running Experiments](#running-experiments)
4. [Prompt Templates](#prompt-templates)
5. [Agent Configurations](#agent-configurations)
6. [Dataset Files](#dataset-files)
7. [Expected Runtime and Cost](#expected-runtime-and-cost)

---

## Environment Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/), Docker (required for SWE-bench and TerminalBench)

```bash
# 1. Clone the repository
git clone https://github.com/ybkim95/agent-scaling.git
cd agent-scaling

# 2. Install dependencies
uv sync --prerelease=allow

# 3. Install flash-attn (required for BrowseComp-Plus environment)
uv pip install --no-build-isolation flash-attn

# 4. Activate the virtual environment
source .venv/bin/activate

# 5. Create a .env file with your API keys (see section below)
cp .env.example .env   # then fill in your keys
```

### Optional: LangFuse Tracing

If you wish to enable LLM call tracing via LangFuse, add the following to your `.env`:

```bash
LANGFUSE_HOST="https://us.cloud.langfuse.com"
LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
```

Then pass `log_langfuse=true` when running experiments.

---

## API Keys Required

Add the following to a `.env` file in the repository root. At minimum, one LLM provider key is required per experiment.

```bash
# LLM providers (add keys for the models you intend to use)
OPENAI_API_KEY="your-openai-key"          # for gpt-5, gpt-5-mini, gpt-5-nano
GEMINI_API_KEY="your-gemini-key"          # for gemini-2.0-flash, gemini-2.5-pro

# Required for BrowseComp-Plus web-search environment
TAVILY_API_KEY="your-tavily-key"

# Optional: LangFuse tracing
LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

**Do NOT commit your `.env` file.** It is listed in `.gitignore`.

---

## Running Experiments

The framework uses [Hydra](https://hydra.cc/docs/intro/) for configuration. The main entry point is:

```bash
python run_scripts/run_experiment.py [overrides...]
```

### Core parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `agent` | Agent configuration name | `multi-agent-centralized` |
| `dataset` | Dataset configuration name | `plancraft-test` |
| `llm.model` | LiteLLM model identifier | `gemini/gemini-2.0-flash` |
| `llm.params.temperature` | Sampling temperature | `0.0` |
| `num_workers` | Parallel instance workers | `1` |
| `max_instances` | Cap on instances to process | unlimited |
| `debug` | Debug mode (3 instances) | `false` |

### Paper experiments

All experiments in the paper use `temperature=0.0` and `n_base_agents=3` (for multi-agent configs).

#### Single-agent baseline

```bash
# PlanCraft
python run_scripts/run_experiment.py agent=single-agent dataset=plancraft-test llm.model=gemini/gemini-2.0-flash

# BrowseComp-Plus
python run_scripts/run_experiment.py agent=single-agent dataset=browsecomp-plus llm.model=openai/gpt-5-mini

# SWE-bench Verified (20-instance subset)
python run_scripts/run_experiment.py agent=single-agent dataset=swebench-verified llm.model=openai/gpt-5-mini

# TerminalBench (20-instance subset)
python run_scripts/run_experiment.py agent=single-agent dataset=terminalbench llm.model=openai/gpt-5-mini
```

#### Multi-agent centralized (lead + subagents)

```bash
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=plancraft-test llm.model=gemini/gemini-2.0-flash
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=browsecomp-plus llm.model=openai/gpt-5-mini
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=swebench-verified llm.model=openai/gpt-5-mini
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=terminalbench llm.model=openai/gpt-5-mini
```

#### Multi-agent decentralized (peer consensus)

```bash
python run_scripts/run_experiment.py agent=multi-agent-decentralized dataset=plancraft-test llm.model=gemini/gemini-2.0-flash
```

#### Multi-agent hybrid

```bash
python run_scripts/run_experiment.py agent=multi-agent-hybrid dataset=plancraft-test llm.model=gemini/gemini-2.0-flash
```

#### Multi-agent independent (no coordination)

```bash
python run_scripts/run_experiment.py agent=multi-agent-independent dataset=plancraft-test llm.model=gemini/gemini-2.0-flash
```

### Output location

Results are saved to:

```
exp_outputs/{dataset_id}/{agent_name}/{llm_provider}/{llm_model}/{date}/{time}/
├── run_config.yaml          # Full resolved configuration
├── run.log                  # Execution log
├── dataset_eval_metrics.json  # Aggregated metrics
└── instance_runs/
    ├── 0000/
    │   └── instance_save.yaml  # Per-instance result
    ├── 0001/
    ...
```

---

## Prompt Templates

All prompt templates are under `prompts/`. They use `{{variable}}` placeholder syntax resolved at runtime.

### Agent prompts

| File | Description |
|------|-------------|
| `prompts/single-agent/single-agent.yaml` | System + user prompt for single-agent runs |
| `prompts/multi-agent/lead_agent.yaml` | Lead agent prompt (centralized / hybrid) |
| `prompts/multi-agent/subagent.yaml` | Subagent prompt (centralized / hybrid / independent) |
| `prompts/multi-agent/agent_decision.yaml` | Agent action-decision prompt |
| `prompts/multi-agent/agent_feedback.yaml` | Inter-agent feedback prompt |
| `prompts/multi-agent/orchestration_decision.yaml` | Orchestration routing prompt |
| `prompts/direct-prompt/direct-prompt.yaml` | Zero-shot direct-prompt baseline |

### Dataset-specific task templates

| File | Dataset |
|------|---------|
| `prompts/dataset-shared/plancraft.yaml` | PlanCraft |
| `prompts/dataset-shared/browsecomp.yaml` | BrowseComp-Plus |
| `prompts/dataset-shared/swebench.yaml` | SWE-bench Verified / SWE-bench Pro |
| `prompts/dataset-shared/terminalbench.yaml` | TerminalBench |
| `prompts/dataset-shared/simpleqa.yaml` | SimpleQA |
| `prompts/dataset-shared/multiagent_simpleqa.yaml` | SimpleQA (multi-agent) |

### Evaluation / grading prompts

| File | Description |
|------|-------------|
| `prompts/eval/grader.yaml` | General LLM-as-judge grader |
| `prompts/eval/qa-grader.yaml` | QA-specific grader |
| `prompts/eval/browsecomp-grader.yaml` | BrowseComp-Plus answer judge |

---

## Agent Configurations

All agent configs are under `run_conf/agent/`.

| File | Agent type | `n_base_agents` | Notes |
|------|-----------|-----------------|-------|
| `run_conf/agent/single-agent.yaml` | Single agent | 1 | Baseline |
| `run_conf/agent/multi-agent-centralized.yaml` | Centralized MAS | 3 | Lead + subagents, orchestrated communication |
| `run_conf/agent/multi-agent-decentralized.yaml` | Decentralized MAS | 3 | Peer consensus, 70% agreement threshold |
| `run_conf/agent/multi-agent-hybrid.yaml` | Hybrid MAS | 3 | Lead + peer communication enabled |
| `run_conf/agent/multi-agent-independent.yaml` | Independent MAS | 3 | No inter-agent coordination |
| `run_conf/agent/direct-prompt.yaml` | Direct prompt | 1 | No iterative tool use |
| `run_conf/agent/multi-agent-research.yaml` | Research MAS | varies | Extended research variant |

Key shared parameters (centralized / decentralized / hybrid / independent):

```yaml
n_base_agents: 3
min_iterations_per_agent: 3
max_iterations_per_agent: 25
max_rounds: 10
```

---

## Dataset Files

All dataset files used in the paper are under `datasets/` and `run_conf/dataset/`.

| Dataset | Config file | Data file | Instances used | Selection method |
|---------|-------------|-----------|----------------|-----------------|
| PlanCraft | `run_conf/dataset/plancraft-test.yaml` | `datasets/plancraft-test.json` | 100 (full test set) | All instances |
| BrowseComp-Plus | `run_conf/dataset/browsecomp-plus.yaml` | `datasets/browsecomp_plus_sampled_100.json` | 100 | Random sample |
| SWE-bench Verified | `run_conf/dataset/swebench-verified.yaml` | `datasets/swebench-verified.json` | 20 | Deterministic shuffle, seed 42, first 20 |
| SWE-bench Pro | `run_conf/dataset/swebench-pro.yaml` | `datasets/swebench-pro.json` | 20 | Deterministic shuffle, seed 42, first 20 |
| TerminalBench | `run_conf/dataset/terminalbench.yaml` | `datasets/terminalbench.json` | 20 | First 20 instances |

See `DATA_AVAILABILITY.md` for full dataset descriptions and source URLs.

---

## Expected Runtime and Cost

All estimates assume `num_workers=1`, `temperature=0.0`, `n_base_agents=3`.

| Dataset | Agent type | Model | Est. wall time | Est. API cost |
|---------|-----------|-------|---------------|---------------|
| PlanCraft (100 inst.) | Single-agent | gemini-2.0-flash | ~1 hr | ~$2 |
| PlanCraft (100 inst.) | Multi-agent-centralized | gemini-2.0-flash | ~3 hr | ~$8 |
| BrowseComp-Plus (100 inst.) | Single-agent | gpt-5-mini | ~2 hr | ~$5 |
| BrowseComp-Plus (100 inst.) | Multi-agent-centralized | gpt-5-mini | ~6 hr | ~$18 |
| SWE-bench Verified (20 inst.) | Single-agent | gpt-5-mini | ~1 hr | ~$4 |
| SWE-bench Verified (20 inst.) | Multi-agent-centralized | gpt-5-mini | ~3 hr | ~$14 |
| TerminalBench (20 inst.) | Single-agent | gpt-5-mini | ~1 hr | ~$4 |
| TerminalBench (20 inst.) | Multi-agent-centralized | gpt-5-mini | ~3 hr | ~$14 |

**Notes:**
- Estimates are approximate; actual cost depends on task difficulty and model verbosity.
- Using `num_workers=4` reduces wall time by ~3-4x with proportional cost.
- SWE-bench and TerminalBench require Docker and pull benchmark container images on first run (~5–10 min overhead).
- Cost estimates use pricing at time of submission; check current provider pricing before large runs.
