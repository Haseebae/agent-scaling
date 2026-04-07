# Data Availability Statement

This document lists all benchmarks used in the paper, their sources, and the subsets selected for evaluation.

---

## Benchmarks

### 1. PlanCraft

- **Reference:** Dagan et al., "PlanCraft: An Open-Ended Multi-Modal Planning Benchmark in Minecraft" (2024)
- **URL:** https://github.com/gautierdag/plancraft
- **Task type:** Tool-use planning (Minecraft crafting)
- **Total instances:** 100 (full test split)
- **Instances used:** 100
- **Subset selection:** Full test set; no sub-sampling
- **Local data file:** `datasets/plancraft-test.json`
- **Dataset config:** `run_conf/dataset/plancraft-test.yaml`

### 2. BrowseComp-Plus

- **Reference:** Extended version of BrowseComp (OpenAI, 2025)
- **URL:** https://github.com/wbbbbbz/BrowseComp-Plus (see also `etc/BrowseComp-Plus/`)
- **Task type:** Multi-hop web-search question answering
- **Total instances:** 100 (sampled)
- **Instances used:** 100
- **Subset selection:** 100-instance random sample from the full BrowseComp-Plus dataset
- **Local data file:** `datasets/browsecomp_plus_sampled_100.json`
- **Dataset config:** `run_conf/dataset/browsecomp-plus.yaml`

### 3. SWE-bench Verified

- **Reference:** Jimenez et al., "SWE-bench: Can Language Models Resolve Real-world GitHub Issues?" (2024)
- **URL:** https://www.swebench.com/
- **Task type:** Software engineering — real GitHub issue resolution in Docker sandboxes
- **Total instances:** 500 (verified split)
- **Instances used:** 20
- **Subset selection:** Deterministic shuffle with `seed=42`, first 20 instances taken
- **Local data file:** `datasets/swebench-verified.json`
- **Dataset config:** `run_conf/dataset/swebench-verified.yaml`
- **Environment:** Requires Docker; benchmark containers pulled automatically on first run

### 4. SWE-bench Pro

- **Reference:** Extended / harder variant of SWE-bench
- **URL:** https://www.swebench.com/
- **Task type:** Software engineering — harder real GitHub issue resolution
- **Total instances:** varies
- **Instances used:** 20
- **Subset selection:** Deterministic shuffle with `seed=42`, first 20 instances taken
- **Local data file:** `datasets/swebench-pro.json`
- **Dataset config:** `run_conf/dataset/swebench-pro.yaml`
- **Environment:** Requires Docker; same container infrastructure as SWE-bench Verified

### 5. TerminalBench

- **Reference:** TerminalBench (2025)
- **URL:** https://www.tbench.ai/
- **Task type:** Terminal / CLI task completion in sandboxed Docker environments
- **Total instances:** 86
- **Instances used:** 20
- **Subset selection:** First 20 instances (instances are ordered by difficulty)
- **Local data file:** `datasets/terminalbench.json`
- **Dataset config:** `run_conf/dataset/terminalbench.yaml`
- **Environment:** Requires Docker

### 6. Finance-Agent / Workbench

- **Reference:** Domain-specific agentic reasoning benchmarks (see paper for full citation)
- **Task type:** Financial reasoning and multi-step tool-use
- **Notes:** Data not redistributed in this repository; see original paper for access instructions

---

## Subset Selection Details

| Dataset | Seed | Selection method |
|---------|------|-----------------|
| PlanCraft | N/A | Full test set |
| BrowseComp-Plus | N/A | 100-instance random sample (fixed) |
| SWE-bench Verified | 42 | `random.shuffle(instances, seed=42)`, first 20 |
| SWE-bench Pro | 42 | `random.shuffle(instances, seed=42)`, first 20 |
| TerminalBench | N/A | First 20 instances in canonical order |

---

## Licensing

- **PlanCraft:** MIT License
- **BrowseComp-Plus:** Apache 2.0 (see `etc/BrowseComp-Plus/`)
- **SWE-bench Verified / Pro:** MIT License
- **TerminalBench:** See https://www.tbench.ai/ for license terms

---

## Reproducibility Note

All dataset JSON files included in this repository (`datasets/`) are either (a) the full benchmark test splits or (b) fixed random samples committed to ensure exact reproducibility. Running experiments with the provided configs will use exactly these instances.

For benchmarks requiring Docker (SWE-bench, TerminalBench), Docker images are pulled automatically from public registries on first run. No additional data download steps are required beyond cloning this repository.
