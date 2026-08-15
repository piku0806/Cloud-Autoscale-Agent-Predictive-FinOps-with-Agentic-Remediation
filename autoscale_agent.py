"""
Predictive Autoscaling + Agentic Remediation Loop
===================================================
Project 1 from the FinOps ML roadmap:
  1. EDA on cloud VM telemetry
  2. Baseline classifier predicting scale_up / scale_down / no_action
  3. Handle class imbalance
  4. A lightweight "agent loop" that consumes predictions and decides
     whether to auto-act, or escalate to a human, with a logged rationale.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

pd.set_option("display.width", 120)

# ---------------------------------------------------------------------
# 1. LOAD + EDA
# ---------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Cloud_Dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
print("=" * 70)
print("DATA OVERVIEW")
print("=" * 70)
print(f"Rows: {len(df)}  |  Time span: {df.timestamp.min()} -> {df.timestamp.max()}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
print(f"\nTarget distribution (%):\n{(df['target'].value_counts(normalize=True) * 100).round(1)}")

# quick correlation of numeric features with each other (sanity check)
numeric_cols = ["cpu_usage", "memory_usage", "net_io", "disk_io", "vCPU",
                 "RAM_GB", "price_per_hour", "latency_ms", "throughput", "cost", "utilization"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
df["target"].value_counts().plot(kind="bar", ax=axes[0], color=["#4FD1C5", "#E0A94D", "#E8705D"])
axes[0].set_title("Target class balance")
axes[0].set_xlabel("")
sub = df.sample(min(300, len(df)), random_state=42)
colors = sub["target"].map({"no_action": "#4FD1C5", "scale_up": "#E0A94D", "scale_down": "#E8705D"})
axes[1].scatter(sub["cpu_usage"], sub["utilization"], c=colors, alpha=0.6)
axes[1].set_xlabel("cpu_usage")
axes[1].set_ylabel("utilization")
axes[1].set_title("CPU vs Utilization by target class")
plt.tight_layout()
plt.savefig("{}/eda_overview.png".format(OUTPUT_DIR), dpi=120)
print("\nSaved EDA plot -> eda_overview.png")

# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek

cat_cols = ["cloud_provider", "region", "vm_type"]
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c + "_enc"] = le.fit_transform(df[c])
    le_dict[c] = le

feature_cols = numeric_cols + ["hour", "dayofweek"] + [c + "_enc" for c in cat_cols]
X = df[feature_cols]
y = df["target"]

target_le = LabelEncoder()
y_enc = target_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 3. HANDLE CLASS IMBALANCE + TRAIN
# ---------------------------------------------------------------------
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"\nClass weights (balanced): "
      f"{dict(zip(target_le.inverse_transform(classes), weights.round(2)))}")

clf = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight=class_weight_dict,
    random_state=42, n_jobs=-1
)
clf.fit(X_train_s, y_train)

# ---------------------------------------------------------------------
# 4. EVALUATE
# ---------------------------------------------------------------------
y_pred = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=target_le.classes_, zero_division=0))
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="YlOrBr")
ax.set_xticks(range(len(target_le.classes_))); ax.set_xticklabels(target_le.classes_, rotation=30)
ax.set_yticks(range(len(target_le.classes_))); ax.set_yticklabels(target_le.classes_)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("{}/confusion_matrix.png".format(OUTPUT_DIR), dpi=120)
print("Saved confusion matrix -> confusion_matrix.png")

feat_imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nTop 5 features:\n{feat_imp.head(5)}")

# ---------------------------------------------------------------------
# 5. AGENTIC REMEDIATION LOOP
# ---------------------------------------------------------------------
# This simulates the "agent" layer: for each new observation, the agent
# takes the model's prediction + confidence, decides whether to
# auto-execute, log-only, or escalate to a human, and writes a
# rationale. In production, the "execute_action" and "notify_human"
# functions would call real cloud APIs / Slack / PagerDuty.

CONFIDENCE_AUTO_THRESHOLD = 0.75   # auto-act only if model is confident
HIGH_COST_PROVIDERS = {"AWS", "Azure", "GCP"}  # placeholder for cost-tier logic


class AutoscaleAgent:
    def __init__(self, model, scaler, target_encoder, feature_cols):
        self.model = model
        self.scaler = scaler
        self.target_encoder = target_encoder
        self.feature_cols = feature_cols
        self.action_log = []

    def decide(self, row: pd.Series) -> dict:
        x = row[self.feature_cols].values.reshape(1, -1)
        x_s = self.scaler.transform(x)
        proba = self.model.predict_proba(x_s)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.target_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        decision = self._policy(pred_label, confidence, row)
        record = {
            "timestamp": row.get("timestamp"),
            "vm_type": row.get("vm_type"),
            "cloud_provider": row.get("cloud_provider"),
            "predicted_action": pred_label,
            "confidence": round(confidence, 3),
            "decision": decision["decision"],
            "rationale": decision["rationale"],
        }
        self.action_log.append(record)
        return record

    def _policy(self, pred_label, confidence, row) -> dict:
        # Guardrail 1: no_action never needs escalation
        if pred_label == "no_action":
            return {"decision": "NO_OP", "rationale": f"Model confidence {confidence:.0%}; system nominal."}

        # Guardrail 2: low confidence -> escalate to human regardless of action
        if confidence < CONFIDENCE_AUTO_THRESHOLD:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "rationale": (f"Predicted '{pred_label}' but confidence {confidence:.0%} "
                              f"is below the {CONFIDENCE_AUTO_THRESHOLD:.0%} auto-act threshold.")
            }

        # Guardrail 3: scale_down on a high-utilization VM is risky -> escalate
        if pred_label == "scale_down" and row.get("utilization", 0) > 70:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "rationale": (f"Model recommends scale_down but observed utilization "
                              f"({row.get('utilization'):.0f}%) looks high — possible SLA risk, "
                              f"deferring to human review.")
            }

        # Otherwise: auto-execute
        return {
            "decision": f"AUTO_EXECUTE:{pred_label}",
            "rationale": f"Confidence {confidence:.0%} clears threshold and guardrails passed."
        }

    def execute_action(self, record):
        """Placeholder for a real cloud API call (resize/terminate/launch)."""
        if record["decision"].startswith("AUTO_EXECUTE"):
            print(f"  [ACTION] Would call cloud API to '{record['predicted_action']}' "
                  f"on a {record['vm_type']} ({record['cloud_provider']}) instance.")
        elif record["decision"] == "ESCALATE_TO_HUMAN":
            print(f"  [ALERT]  Would notify on-call: {record['rationale']}")


agent = AutoscaleAgent(clf, scaler, target_le, feature_cols)

print("\n" + "=" * 70)
print("AGENT LOOP — sample run on 10 held-out observations")
print("=" * 70)
sample = df.loc[X_test.index].sample(10, random_state=7)
for _, row in sample.iterrows():
    record = agent.decide(row)
    print(f"{row['timestamp']} | {row['vm_type']:>14} | pred={record['predicted_action']:<10} "
          f"conf={record['confidence']:.0%} -> {record['decision']}")
    agent.execute_action(record)

log_df = pd.DataFrame(agent.action_log)
log_df.to_csv("{}/agent_decision_log.csv".format(OUTPUT_DIR), index=False)
print("\nFull decision log saved -> agent_decision_log.csv")

decision_summary = log_df["decision"].apply(lambda d: d.split(":")[0]).value_counts()
print(f"\nDecision summary across sample run:\n{decision_summary}")
