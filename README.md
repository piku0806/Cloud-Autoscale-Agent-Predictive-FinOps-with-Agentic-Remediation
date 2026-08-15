# Cloud Autoscale Agent — Predictive FinOps with Agentic Remediation

A machine learning system that predicts whether a cloud VM needs to **scale up**, **scale down**, or take **no action**, wrapped in a lightweight autonomous agent that decides whether to act automatically, log the event, or escalate to a human — with every decision logged and explained.

This project sits at the intersection of **FinOps** (cloud cost management) and **agentic AI**: it doesn't just predict — it *acts*, within guardrails.

---

## Why this exists

Most cloud cost/scaling ML demos stop at "here's a model and its accuracy." This project goes one step further and asks: *what happens after the prediction?* In a real FinOps setting, a model's output is only useful if something acts on it — safely, explainably, and with a human in the loop when the stakes are high.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  VM Metrics  │ ──▶ │ RandomForest      │ ──▶ │  AutoscaleAgent      │
│  (telemetry) │     │ Classifier        │     │  policy + guardrails │
└─────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                          │
                                     ┌────────────────────┼────────────────────┐
                                     ▼                    ▼                    ▼
                                  NO_OP           AUTO_EXECUTE          ESCALATE_TO_HUMAN
                             (system nominal)  (confident + safe)   (low confidence OR
                                                                      risky scale-down)
```

## Dataset

`data/Cloud_Dataset.csv` — 1,000 rows of synthetic VM telemetry at 5-minute intervals across AWS, Azure, and GCP.

| Column | Description |
|---|---|
| `timestamp` | Observation time |
| `cpu_usage`, `memory_usage`, `net_io`, `disk_io` | Resource utilization metrics |
| `cloud_provider`, `region`, `vm_type` | Infrastructure metadata |
| `vCPU`, `RAM_GB`, `price_per_hour` | Instance specs and pricing |
| `latency_ms`, `throughput`, `cost`, `utilization` | Performance and cost metrics |
| `target` | Label: `scale_up` / `scale_down` / `no_action` |

> **Note:** This is a synthetic/demo dataset. The near-perfect model accuracy below reflects that the labels are likely rule-derived from the input features rather than messy real-world ops decisions — expect lower accuracy and heavier feature engineering (rolling windows, rate-of-change features) on real production telemetry.

## What's inside

1. **EDA** — class balance, feature distributions, quick visual sanity checks
2. **Baseline model** — Random Forest classifier with balanced class weights to handle the imbalanced target (65.8% no_action / 29.3% scale_up / 4.9% scale_down)
3. **Agent loop** (`AutoscaleAgent`) — consumes model predictions and applies a decision policy:
   - **NO_OP** — model predicts no action needed
   - **AUTO_EXECUTE** — confident prediction (≥75%) that clears all guardrails
   - **ESCALATE_TO_HUMAN** — triggered by low model confidence, *or* by a specific safety guardrail that blocks auto-approving a `scale_down` when observed utilization is still above 70% (possible SLA risk)
4. **Decision log** — every agent decision is written to CSV with a plain-English rationale, for auditability

## Results

- **Macro F1:** 0.978
- Precision/recall ≥ 0.92 across all three classes, including the rare `scale_down` class
- Top predictive features: `utilization`, `memory_usage`, `cpu_usage`, `throughput`

See `outputs/confusion_matrix.png` and `outputs/eda_overview.png` for visuals, and `outputs/agent_decision_log.csv` for a sample agent run.

## Project structure

```
.
├── data/
│   └── Cloud_Dataset.csv       # Source telemetry data
├── src/
│   └── autoscale_agent.py      # EDA + model training + agent loop (main entry point)
├── outputs/
│   ├── eda_overview.png
│   ├── confusion_matrix.png
│   └── agent_decision_log.csv
├── notebooks/                  # (optional) exploratory notebooks go here
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/cloud-autoscale-agent.git
cd cloud-autoscale-agent
pip install -r requirements.txt
```

## Usage

```bash
python src/autoscale_agent.py
```

This will:
1. Print an EDA summary and save `outputs/eda_overview.png`
2. Train the classifier and print a classification report
3. Save `outputs/confusion_matrix.png`
4. Run the agent against 10 sample held-out observations and save `outputs/agent_decision_log.csv`

## Roadmap / next steps

- [ ] Replace `execute_action`'s print statements with real cloud SDK calls (boto3 / azure-mgmt / google-cloud-compute)
- [ ] Route `ESCALATE_TO_HUMAN` decisions to Slack or PagerDuty instead of stdout
- [ ] Add rolling-window and rate-of-change features for use on real (non-synthetic) telemetry
- [ ] Add a second "SLA-guardian" agent that can veto the autoscale agent's decisions (multi-agent negotiation)
- [ ] Swap the static confidence threshold for a calibrated, cost-aware decision threshold

## License

MIT — see [LICENSE](LICENSE).
