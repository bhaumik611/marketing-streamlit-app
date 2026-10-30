"""Interactive Streamlit Web Application for UCI Bank Marketing Classifier.

Built with a custom NumPy-based logistic regression model (no scikit-learn).
Supports instant loading of the pre-trained model (.npz/.nps), local bank-full.csv dataset,
interactive single-client predictions, batch CSV scoring, and model diagnostics.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Add src folder to sys.path so modules resolve whether run from root or src
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bank_marketing_core import (
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    load_pipeline,
    predict_one,
    predict_batch,
    download_bank_data,
)


# --- Page Configuration ---
st.set_page_config(
    page_title="Bank Term Deposit Predictor | ML Demo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .card-header {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .card-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .prediction-box-yes {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 16px;
        border-radius: 6px;
        margin: 12px 0;
    }
    .prediction-box-no {
        background-color: #F3F4F6;
        border-left: 5px solid #6B7280;
        padding: 16px;
        border-radius: 6px;
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Cached Resources ---
@st.cache_resource(show_spinner="Loading trained model pipeline...")
def get_pipeline():
    """Load the pre-trained model and preprocessor."""
    return load_pipeline()


@st.cache_data(show_spinner="Loading bank-full.csv dataset...")
def get_dataset():
    """Load the local bank dataset."""
    try:
        return download_bank_data()
    except Exception as e:
        st.warning(f"Could not load bank-full.csv automatically: {e}")
        return None


# Load Pipeline
try:
    pipeline = get_pipeline()
    pipeline_loaded = True
except Exception as exc:
    st.error(f"Error loading trained model pipeline: {exc}")
    pipeline_loaded = False
    pipeline = None


# --- Preset Client Profiles ---
PRESETS = {
    "👤 Typical / Average Client": {
        "age": 38,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1400,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
    },
    "🚀 High-Propensity Client (Likely to Subscribe)": {
        "age": 25,
        "job": "student",
        "marital": "single",
        "education": "tertiary",
        "default": "no",
        "balance": 4500,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 12,
        "month": "mar",
    },
    "🔻 Low-Propensity Client (Unlikely to Subscribe)": {
        "age": 42,
        "job": "blue-collar",
        "marital": "married",
        "education": "primary",
        "default": "yes",
        "balance": 50,
        "housing": "yes",
        "loan": "yes",
        "contact": "unknown",
        "day": 20,
        "month": "jul",
    },
    "💼 Retired Investor": {
        "age": 68,
        "job": "retired",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 8200,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 18,
        "month": "oct",
    },
}


# Initialize Session State for inputs
if "client_data" not in st.session_state:
    st.session_state.client_data = PRESETS["👤 Typical / Average Client"].copy()


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    if pipeline_loaded:
        model_filename = Path(pipeline.get("model_path", "scratch_logistic_model.npz")).name
        st.success(f"🟢 **Trained Model Loaded:** `{model_filename}`")
        default_threshold = float(pipeline.get("threshold", 0.16))
    else:
        st.warning("⚠️ Using fallback configuration")
        default_threshold = 0.16

    st.markdown("---")
    st.subheader("🎯 Decision Threshold Tuning")
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.05,
        max_value=0.90,
        value=default_threshold,
        step=0.01,
        help=(
            "The model outputs a probability between 0 and 1. If probability >= threshold, "
            "the client is classified as 'Yes (Subscribe)'. The optimal threshold selected "
            f"via validation F1 score is {default_threshold:.2f} due to class imbalance."
        ),
    )
    if threshold != default_threshold:
        if st.button("🔄 Reset to Optimal Threshold"):
            st.session_state.threshold = default_threshold
            st.rerun()

    st.markdown("---")
    st.subheader("⚡ Quick Load Test Profiles")
    preset_choice = st.selectbox(
        "Select a pre-filled persona:",
        list(PRESETS.keys()),
        index=0,
    )
    if st.button("Apply Profile"):
        st.session_state.client_data = PRESETS[preset_choice].copy()
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        **Architecture & Tech Stack:**
        - **Algorithm**: Binary Logistic Regression
        - **Implementation**: Pure NumPy gradient descent (Scratch)
        - **Preprocessing**: Training-only standard scaling & one-hot encoding
        - **Dataset**: UCI Bank Marketing (45,211 instances)
        """
    )


# --- Header ---
st.markdown('<div class="main-title">🏦 Bank Term Deposit Propensity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An end-to-end Machine Learning web application built from scratch '
    'using NumPy and Pandas to prioritize pre-contact telemarketing prospects.</div>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ About the Project & Historical Context", expanded=False):
    st.markdown(
        """
        - **Objective**: Predict whether a bank customer will subscribe to a term deposit (`yes` / `no`) before a telemarketing call is initiated.
        - **Pre-contact Scope**: To prevent data leakage and simulate realistic pre-campaign targeting, post-call variables (such as `duration`) and historical campaign counters (`campaign`, `pdays`, `previous`, `poutcome`) are excluded. Only the 11 pre-contact attributes are used.
        - **Class Imbalance**: In the full dataset of 45,211 calls, only ~11.7% of clients subscribed. A standard 0.50 threshold yields high nominal accuracy but misses most prospective clients. We tune the threshold to **0.16** based on validation F1 score.
        - **Dataset Citation**: Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, Elsevier.
        """
    )

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Individual Prediction",
    "📊 Model Performance & Metrics",
    "📁 Batch Prediction (CSV)",
    "🔍 Dataset Explorer & Insights",
])


# ==============================================================================
# TAB 1: INDIVIDUAL PREDICTION
# ==============================================================================
with tab1:
    st.subheader("Enter Client Information (Pre-Contact)")
    st.caption("Adjust client attributes below and click **Estimate Subscription Propensity**.")

    cur_data = st.session_state.client_data

    # Form with 3 Columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Demographics")
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=int(cur_data.get("age", 38)), step=1
        )
        job_options = [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"
        ]
        job = st.selectbox(
            "Job Title", job_options,
            index=job_options.index(cur_data.get("job", "technician")) if cur_data.get("job") in job_options else 0
        )
        marital_options = ["married", "single", "divorced"]
        marital = st.selectbox(
            "Marital Status", marital_options,
            index=marital_options.index(cur_data.get("marital", "married")) if cur_data.get("marital") in marital_options else 0
        )
        education_options = ["secondary", "tertiary", "primary", "unknown"]
        education = st.selectbox(
            "Education Level", education_options,
            index=education_options.index(cur_data.get("education", "secondary")) if cur_data.get("education") in education_options else 0
        )

    with col2:
        st.markdown("##### 💳 Financial Profile")
        default_opts = ["no", "yes"]
        default = st.selectbox(
            "Has Credit in Default?", default_opts,
            index=default_opts.index(cur_data.get("default", "no")) if cur_data.get("default") in default_opts else 0
        )
        balance = st.number_input(
            "Average Yearly Balance (€)", value=int(cur_data.get("balance", 1400)), step=100,
            help="Average yearly balance in euros."
        )
        housing_opts = ["yes", "no"]
        housing = st.selectbox(
            "Has Housing Loan?", housing_opts,
            index=housing_opts.index(cur_data.get("housing", "yes")) if cur_data.get("housing") in housing_opts else 0
        )
        loan_opts = ["no", "yes"]
        loan = st.selectbox(
            "Has Personal Loan?", loan_opts,
            index=loan_opts.index(cur_data.get("loan", "no")) if cur_data.get("loan") in loan_opts else 0
        )

    with col3:
        st.markdown("##### 📞 Planned Outreach")
        contact_opts = ["cellular", "telephone", "unknown"]
        contact = st.selectbox(
            "Contact Communication Type", contact_opts,
            index=contact_opts.index(cur_data.get("contact", "cellular")) if cur_data.get("contact") in contact_opts else 0
        )
        month_opts = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month = st.selectbox(
            "Outreach Month", month_opts,
            index=month_opts.index(cur_data.get("month", "may")) if cur_data.get("month") in month_opts else 4
        )
        day = st.number_input(
            "Day of Month", min_value=1, max_value=31, value=int(cur_data.get("day", 15)), step=1
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Estimate Subscription Propensity", type="primary", use_container_width=True)

    # Process Prediction
    input_values = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
    }

    if predict_clicked or True:  # Compute live prediction
        if pipeline is not None:
            # Overwrite threshold from sidebar
            pipeline_copy = dict(pipeline)
            pipeline_copy["threshold"] = threshold
            prediction = predict_one(pipeline_copy, input_values)

            prob = prediction["subscription_probability"]
            is_sub = prediction["predicted_subscription"]
            baseline_prob = 0.117  # ~11.7% historical positive rate

            st.markdown("---")
            st.subheader("🎯 Prediction Results")

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric(
                    label="Estimated Subscription Probability",
                    value=f"{prob:.1%}",
                    delta=f"{(prob - baseline_prob):+.1%} vs dataset baseline (11.7%)",
                )
                
                # Progress Bar relative to threshold
                st.caption(f"Decision Threshold: **{threshold:.2f}**")
                st.progress(min(max(float(prob), 0.0), 1.0))

            with res_col2:
                if is_sub:
                    st.markdown(
                        f"""
                        <div class="prediction-box-yes">
                            <h3 style="color:#065F46; margin:0;">✅ Likely to Subscribe (Target for Outreach)</h3>
                            <p style="margin:4px 0 0 0; color:#047857;">
                                The predicted probability (<strong>{prob:.1%}</strong>) exceeds the calibrated 
                                decision threshold of <strong>{threshold:.2f}</strong>. Prioritizing this contact is recommended.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-box-no">
                            <h3 style="color:#374151; margin:0;">❌ Unlikely to Subscribe (Low Priority)</h3>
                            <p style="margin:4px 0 0 0; color:#4B5563;">
                                The predicted probability (<strong>{prob:.1%}</strong>) is below the decision 
                                threshold of <strong>{threshold:.2f}</strong>. Telemarketing outreach is not prioritized for this profile.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Feature Impact / Interpretability
            contributions = prediction.get("contributions", [])
            if contributions:
                st.markdown("##### 🔍 Key Feature Influences on This Prediction")
                st.caption(
                    "The custom scratch logistic regression model weights indicate which attributes pushed "
                    "the client's score up (positive log-odds) or down (negative log-odds):"
                )
                
                contrib_cols = st.columns(2)
                positives = [c for c in contributions if c[1] > 0][:4]
                negatives = [c for c in contributions if c[1] < 0][:4]

                with contrib_cols[0]:
                    st.markdown("**🟢 Factors Increasing Subscription Odds:**")
                    if positives:
                        for feat, imp in positives:
                            st.write(f"- `{feat}`: **+{imp:.2f}**")
                    else:
                        st.write("*(None of the strong positive features were active)*")

                with contrib_cols[1]:
                    st.markdown("**🔴 Factors Decreasing Subscription Odds:**")
                    if negatives:
                        for feat, imp in negatives:
                            st.write(f"- `{feat}`: **{imp:.2f}**")
                    else:
                        st.write("*(None of the strong negative features were active)*")


# ==============================================================================
# TAB 2: MODEL PERFORMANCE & METRICS
# ==============================================================================
with tab2:
    st.subheader("📊 Scratch Logistic Regression: Performance & Evaluation")
    st.caption(
        "All scores below were measured on held-out data using the custom NumPy model "
        "trained with full-batch gradient descent."
    )

    if pipeline is not None and "summary" in pipeline:
        summary = pipeline["summary"]
        test_metrics = summary.get("test_metrics", {})
        split_metrics = summary.get("split_metrics", {})

        # Top Metric Cards
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        with m_col1:
            st.metric("ROC AUC", f"{test_metrics.get('roc_auc', 0.716):.3f}")
        with m_col2:
            st.metric("PR AUC", f"{test_metrics.get('average_precision', 0.263):.3f}")
        with m_col3:
            st.metric("Accuracy", f"{test_metrics.get('accuracy', 0.784):.1%}")
        with m_col4:
            st.metric("Recall (Sensitivity)", f"{test_metrics.get('recall', 0.492):.1%}")
        with m_col5:
            st.metric("Precision", f"{test_metrics.get('precision', 0.269):.1%}")
        with m_col6:
            st.metric("F1-Score", f"{test_metrics.get('f1', 0.348):.3f}")

        st.markdown("---")

        # Confusion Matrix & Split Comparison
        sec1, sec2 = st.columns([1, 1])

        with sec1:
            st.markdown("##### 🔲 Held-Out Test Confusion Matrix")
            tn = test_metrics.get("tn", 4926)
            fp = test_metrics.get("fp", 1063)
            fn = test_metrics.get("fn", 403)
            tp = test_metrics.get("tp", 391)

            cm_df = pd.DataFrame(
                [
                    [f"TN: {tn:,} (No call, correct)", f"FP: {fp:,} (Wasted call)"],
                    [f"FN: {fn:,} (Missed subscriber)", f"TP: {tp:,} (Captured subscriber)"],
                ],
                index=["Actual: No", "Actual: Yes"],
                columns=["Predicted: No", "Predicted: Yes"],
            )
            st.table(cm_df)

            st.info(
                f"**Business Impact**: At threshold **{test_metrics.get('threshold', 0.16):.2f}**, the model captures "
                f"**{test_metrics.get('recall', 0.492):.1%}** of all prospective depositors (TP = {tp:,}) "
                f"while eliminating **{test_metrics.get('specificity', 0.823):.1%}** of non-interested cold calls (TN = {tn:,})."
            )

        with sec2:
            st.markdown("##### 📈 Performance Across Splits (Overfitting Check)")
            if split_metrics:
                rows = []
                for split_name in ["train", "validation", "test"]:
                    if split_name in split_metrics:
                        m = split_metrics[split_name]
                        rows.append({
                            "Split": split_name.capitalize(),
                            "ROC AUC": f"{m.get('roc_auc', 0):.3f}",
                            "Average Precision": f"{m.get('average_precision', 0):.3f}",
                            "Accuracy": f"{m.get('accuracy', 0):.1%}",
                            "Recall": f"{m.get('recall', 0):.1%}",
                            "Precision": f"{m.get('precision', 0):.1%}",
                            "F1": f"{m.get('f1', 0):.3f}",
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            diag = summary.get("fit_diagnosis", "")
            if diag:
                st.caption(f"**Diagnostic Evaluation**: {diag}")

        st.markdown("---")

        # Learned Coefficients (Top Weights)
        st.markdown("##### ⚖️ Top Model Coefficients (Feature Importance)")
        st.caption("Learned weights from scratch gradient descent. Positive values increase subscription probability.")
        
        coef_df = pipeline.get("coefficients", None)
        if coef_df is not None and not coef_df.empty:
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.markdown("**Top Positive Predictors:**")
                pos_coefs = coef_df[coef_df["coefficient"] > 0].sort_values("coefficient", ascending=False).head(8)
                st.dataframe(pos_coefs[["feature", "coefficient"]], hide_index=True, use_container_width=True)

            with c_col2:
                st.markdown("**Top Negative Predictors:**")
                neg_coefs = coef_df[coef_df["coefficient"] < 0].sort_values("coefficient", ascending=True).head(8)
                st.dataframe(neg_coefs[["feature", "coefficient"]], hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 3: BATCH PREDICTION (CSV)
# ==============================================================================
with tab3:
    st.subheader("📁 Batch Scoring & CSV Evaluation")
    st.caption(
        "Upload client records or load a random test sample directly from `bank-full.csv` "
        "to score multiple prospects simultaneously."
    )

    sample_button = st.button("🎲 Sample 10 Clients from bank-full.csv")

    uploaded_file = st.file_uploader("Or upload your own client CSV file:", type=["csv"])

    batch_df: pd.DataFrame | None = None

    if sample_button:
        dataset = get_dataset()
        if dataset is not None:
            batch_df = dataset.sample(n=10, random_state=int(np.random.randint(1, 10000)))
            st.success("Loaded 10 random samples from `bank-full.csv`.")
    elif uploaded_file is not None:
        try:
            # Check separator (semicolon vs comma)
            content = uploaded_file.getvalue().decode("utf-8")
            sep = ";" if content.count(";") > content.count(",") else ","
            uploaded_file.seek(0)
            batch_df = pd.read_csv(uploaded_file, sep=sep)
            st.success(f"Uploaded CSV with {len(batch_df)} rows.")
        except Exception as err:
            st.error(f"Error reading uploaded CSV: {err}")

    if batch_df is not None and pipeline is not None:
        try:
            # Ensure required features exist
            missing_feats = [f for f in FEATURES if f not in batch_df.columns]
            if missing_feats:
                st.error(f"The input data is missing required feature columns: {missing_feats}")
            else:
                scored_df = predict_batch(pipeline, batch_df, threshold=threshold)

                # Show metrics
                tot = len(scored_df)
                pos_count = int(scored_df["predicted_subscription"].sum())
                st.markdown(
                    f"**Results Summary:** **{pos_count}** out of **{tot}** clients (**{pos_count/tot:.1%}**) "
                    f"predicted to subscribe at threshold **{threshold:.2f}**."
                )

                # Display table
                display_cols = (
                    ["predicted_probability", "prediction_label"]
                    + (["y"] if "y" in scored_df.columns else [])
                    + FEATURES
                )
                st.dataframe(
                    scored_df[display_cols].style.format({"predicted_probability": "{:.1%}"}),
                    use_container_width=True,
                )

                # Download button
                csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Scored CSV",
                    data=csv_bytes,
                    file_name="bank_marketing_predictions.csv",
                    mime="text/csv",
                )
        except Exception as err:
            st.error(f"Batch prediction error: {err}")


# ==============================================================================
# TAB 4: DATASET EXPLORER & INSIGHTS
# ==============================================================================
with tab4:
    st.subheader("🔍 Dataset Exploration (`bank-full.csv`)")
    
    df = get_dataset()
    if df is not None:
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Total Records", f"{len(df):,}")
        with d2:
            sub_count = int((df["y"].astype(str).str.lower() == "yes").sum())
            st.metric("Subscribers (Class 1)", f"{sub_count:,} ({sub_count/len(df):.1%})")
        with d3:
            non_sub = len(df) - sub_count
            st.metric("Non-Subscribers (Class 0)", f"{non_sub:,} ({non_sub/len(df):.1%})")
        with d4:
            st.metric("Pre-Contact Features", f"{len(FEATURES)}")

        st.markdown("---")
        st.markdown("##### 📋 Dataset Preview")
        st.dataframe(df[FEATURES + ["y"]].head(25), use_container_width=True)

        st.markdown("---")
        st.markdown("##### 📊 Subscription Rates by Demographics")
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            st.markdown("**Deposit Subscription Rate by Job:**")
            job_rates = (
                df.assign(is_yes=(df["y"].str.lower() == "yes").astype(int))
                .groupby("job")["is_yes"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(job_rates)

        with g_col2:
            st.markdown("**Deposit Subscription Rate by Education:**")
            edu_rates = (
                df.assign(is_yes=(df["y"].str.lower() == "yes").astype(int))
                .groupby("education")["is_yes"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(edu_rates)
    else:
        st.info("Dataset file `bank-full.csv` not found in project directory.")
