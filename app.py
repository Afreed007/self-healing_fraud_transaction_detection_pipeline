"""
app.py  –  Live Adaptive Fraud Detection Dashboard
---------------------------------------------------
Reads report files from reports/benchmarks/ and renders a
continuously-updating monitoring dashboard.

Run alongside the pipeline:
    streamlit run app.py
"""

import ast
import os
import time

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Fraud Detection Pipeline",
    layout="wide",
    page_icon="🛡️",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tighter metric cards */
  [data-testid="metric-container"] {
      background: #0e1117;
      border: 1px solid #21262d;
      border-radius: 10px;
      padding: 16px 20px;
  }
  [data-testid="metric-container"] label { font-size: 0.78rem; color: #8b949e; }
  [data-testid="metric-container"] [data-testid="metric-value"] {
      font-size: 1.7rem; font-weight: 700;
  }
  /* Section headers */
  h3 { border-bottom: 1px solid #21262d; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Data loader ──────────────────────────────────────────────────────────────

ACTION_LABELS = {
    "0": "Ignore drift",
    "1": "Retrain — new data",
    "2": "Retrain — mixed memory",
    0  : "Ignore drift",
    1  : "Retrain — new data",
    2  : "Retrain — mixed memory",
}


def load_reports() -> list[dict]:
    """
    Parse every batch_N_report.txt and return sorted list of dicts.
    Falls back gracefully if the pipeline hasn't run yet.
    """
    reports_dir = "reports/benchmarks"
    rows = []

    if not os.path.exists(reports_dir):
        return rows

    for fname in sorted(os.listdir(reports_dir)):
        if not (fname.startswith("batch_") and fname.endswith("_report.txt")):
            continue
        try:
            bid = int(fname.split("_")[1])
        except (IndexError, ValueError):
            continue

        try:
            with open(os.path.join(reports_dir, fname)) as f:
                lines = f.readlines()

            rec = {"batch_id": bid, "drift": False, "schema_healed": False,
                   "action": 0, "reward": 0.0, "f1_pre": 0.0, "f1_post": 0.0,
                   "precision": 0.0, "recall": 0.0, "n_records": 0,
                   "ts": "—"}

            for line in lines:
                line = line.strip()
                if line.startswith("Drift:"):
                    rec["drift"] = "True" in line
                elif line.startswith("Schema_Healed:"):
                    rec["schema_healed"] = "True" in line
                elif line.startswith("Action:"):
                    try:
                        rec["action"] = int(line.split(":")[1].strip())
                    except Exception:
                        pass
                elif line.startswith("Reward:"):
                    try:
                        rec["reward"] = float(line.split(":")[1].strip())
                    except Exception:
                        pass
                elif line.startswith("Records:"):
                    try:
                        rec["n_records"] = int(line.split(":")[1].strip())
                    except Exception:
                        pass
                elif line.startswith("Timestamp:"):
                    rec["ts"] = line.split(":", 1)[1].strip()
                elif line.startswith("Old:"):
                    try:
                        d = ast.literal_eval(line.split("Old:")[1].strip())
                        rec["f1_pre"] = d.get("F1 Score", 0.0)
                    except Exception:
                        pass
                elif line.startswith("New:"):
                    try:
                        d = ast.literal_eval(line.split("New:")[1].strip())
                        rec["f1_post"]   = d.get("F1 Score", 0.0)
                        rec["precision"] = d.get("Precision", 0.0)
                        rec["recall"]    = d.get("Recall", 0.0)
                    except Exception:
                        pass

            rows.append(rec)
        except Exception as e:
            st.warning(f"Could not parse {fname}: {e}")

    rows.sort(key=lambda r: r["batch_id"])
    return rows


# ── Main dashboard ───────────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("## 🛡️ Adaptive Fraud Detection Pipeline")
    st.caption("Real-time monitoring · Auto-refreshes every 5 seconds")
    st.divider()

    reports = load_reports()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=48)
        st.markdown("### Pipeline monitor")

        if st.button("🔄 Refresh now"):
            st.rerun()

        refresh = st.toggle("Auto-refresh (5 s)", value=True)
        st.divider()
        st.markdown("**How to run**")
        st.code("python cly.py train-base\npython cly.py run-pipeline\nstreamlit run app.py",
                language="bash")
        st.divider()
        st.markdown("**Architecture**")
        st.markdown("""
- 🟢 **Producer** — streams 500 rec/s  
- 🔵 **Inference** — per-record XGBoost  
- 🟠 **BG Thread** — KS test + RL retrain  
- 🔴 **Dashboard** — live reports  
        """)

    # ── No data yet ───────────────────────────────────────────────────────────
    if not reports:
        st.info("No batch reports found yet. Start the pipeline with "
                "`python cly.py run-pipeline` and reports will appear here.")
        if refresh:
            time.sleep(5)
            st.rerun()
        return

    # ── Summary KPI row ───────────────────────────────────────────────────────
    latest = reports[-1]
    n_batches   = len(reports)
    n_drifted   = sum(1 for r in reports if r["drift"])
    n_healed    = sum(1 for r in reports if r["schema_healed"])
    total_recs  = sum(r["n_records"] for r in reports)
    avg_f1_post = sum(r["f1_post"] for r in reports) / n_batches
    avg_reward  = sum(r["reward"]   for r in reports) / n_batches

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Virtual batches", n_batches)
    c2.metric("Records processed", f"{total_recs:,}")
    c3.metric("Drift events", n_drifted,
              delta=f"{n_drifted/n_batches*100:.0f}% of batches",
              delta_color="inverse")
    c4.metric("Schema heals", n_healed)
    c5.metric("Avg F1 (post)", f"{avg_f1_post:.4f}")
    c6.metric("Avg RL reward", f"{avg_reward:+.4f}")

    st.divider()

    # ── Latest batch detail ───────────────────────────────────────────────────
    st.markdown(f"### 🔍 Latest virtual batch — #{latest['batch_id']}")
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Timestamp", latest["ts"])
    dc2.metric("Records",   f"{latest['n_records']:,}")

    drift_label = "🚨 Detected" if latest["drift"] else "✅ Stable"
    drift_delta = f"F1 before: {latest['f1_pre']:.4f}"
    dc3.metric("Concept drift", drift_label,
               delta=drift_delta,
               delta_color="inverse" if latest["drift"] else "normal")

    action_str = ACTION_LABELS.get(latest["action"], str(latest["action"]))
    dc4.metric("RL action", action_str,
               delta=f"Reward: {latest['reward']:+.4f}")

    # Schema heal badge
    if latest["schema_healed"]:
        st.info("🧠 Schema drift detected in this batch — "
                "LLM healer resolved unknown columns and cached mappings.")

    st.divider()

    # ── F1 trend chart ────────────────────────────────────────────────────────
    st.markdown("### 📈 F1 score — static baseline vs adaptive system")

    bids   = [r["batch_id"]  for r in reports]
    f1_pre = [r["f1_pre"]    for r in reports]
    f1_post= [r["f1_post"]   for r in reports]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bids, y=f1_pre, mode="lines+markers",
        name="Baseline (frozen model)",
        line=dict(color="#f85149", dash="dash", width=2),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=bids, y=f1_post, mode="lines+markers",
        name="Adaptive (RL-healed)",
        line=dict(color="#3fb950", width=3),
        marker=dict(size=9, symbol="circle"),
        fill="tonexty",
        fillcolor="rgba(63,185,80,0.08)",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(title="Virtual batch #", tickvals=bids, gridcolor="#21262d"),
        yaxis=dict(title="F1 score", range=[-0.05, 1.05], gridcolor="#21262d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Dual column charts ────────────────────────────────────────────────────
    left, right = st.columns(2)

    # Reward over batches
    with left:
        st.markdown("#### 🎯 RL agent reward per batch")
        rewards = [r["reward"] for r in reports]
        colors  = ["#3fb950" if v >= 0 else "#f85149" for v in rewards]
        fig2 = go.Figure(go.Bar(
            x=bids, y=rewards,
            marker_color=colors,
            text=[f"{v:+.3f}" for v in rewards],
            textposition="outside",
        ))
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Virtual batch #", tickvals=bids, gridcolor="#21262d"),
            yaxis=dict(title="Reward", gridcolor="#21262d"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Action distribution
    with right:
        st.markdown("#### 🤖 RL action distribution")
        action_counts = {
            "Ignore (0)"        : sum(1 for r in reports if r["action"] == 0),
            "Retrain-New (1)"   : sum(1 for r in reports if r["action"] == 1),
            "Retrain-Mixed (2)" : sum(1 for r in reports if r["action"] == 2),
        }
        fig3 = go.Figure(go.Pie(
            labels=list(action_counts.keys()),
            values=list(action_counts.values()),
            hole=0.55,
            marker_colors=["#388bfd", "#3fb950", "#d29922"],
            textinfo="label+percent",
        ))
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Precision / Recall table ──────────────────────────────────────────────
    st.markdown("### 📋 Per-batch metrics table")
    df_table = pd.DataFrame([{
        "Batch"          : r["batch_id"],
        "Timestamp"      : r["ts"],
        "Records"        : r["n_records"],
        "Schema healed"  : "✅" if r["schema_healed"] else "—",
        "Drift"          : "🚨" if r["drift"] else "✅",
        "RL action"      : ACTION_LABELS.get(r["action"], str(r["action"])),
        "F1 (before)"    : f"{r['f1_pre']:.4f}",
        "F1 (after)"     : f"{r['f1_post']:.4f}",
        "Precision"      : f"{r['precision']:.4f}",
        "Recall"         : f"{r['recall']:.4f}",
        "Reward"         : f"{r['reward']:+.4f}",
    } for r in reports])
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── Drift / schema timeline ───────────────────────────────────────────────
    st.markdown("### 🗂️ Event timeline")
    for r in reversed(reports):
        icon  = "🚨" if r["drift"] else "✅"
        sicon = "🧠" if r["schema_healed"] else "  "
        aname = ACTION_LABELS.get(r["action"], str(r["action"]))
        st.markdown(
            f"`Batch {r['batch_id']:>3}` &nbsp; {icon} drift &nbsp; "
            f"{sicon} schema &nbsp; → &nbsp; **{aname}** &nbsp; "
            f"F1: `{r['f1_pre']:.3f}` → `{r['f1_post']:.3f}` &nbsp; "
            f"reward: `{r['reward']:+.3f}` &nbsp; _{r['ts']}_",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Pipeline: Apache Kafka · XGBoost · KS-Test · Q-Learning RL · LLM Schema Healer")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()