import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Deposit Prediction", page_icon="🏦")
st.title("🏦 Bank Term Deposit Prediction")
st.markdown(
    "Predict whether a client will **subscribe to a term deposit** "
    "based on demographic, campaign, and macroeconomic features. "
    "Trained on the [UCI Bank Marketing Dataset]"
    "(https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)."
)

# Load model
try:
    model = joblib.load("bank_model.pkl")
except Exception:
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

# ── Sidebar: Client ──────────────────────────────────────────────
st.sidebar.header("👤 Client")
age = st.sidebar.slider("Age", 18, 95, 35)
job = st.sidebar.selectbox("Occupation", [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"])
marital = st.sidebar.selectbox("Marital Status", [
    "married", "single", "divorced", "unknown"])
education = st.sidebar.selectbox("Education Level", [
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown"])
default = st.sidebar.selectbox("Credit in Default?", ["no", "yes", "unknown"])
housing = st.sidebar.radio("Housing Loan?", ["yes", "no", "unknown"])
loan = st.sidebar.radio("Personal Loan?", ["yes", "no", "unknown"])

# ── Sidebar: Campaign ────────────────────────────────────────────
st.sidebar.header("📞 Campaign")
contact = st.sidebar.selectbox("Contact Method", ["cellular", "telephone"])
month = st.sidebar.selectbox("Month of Last Contact", [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"])
day_of_week = st.sidebar.selectbox("Day of Last Contact", [
    "mon", "tue", "wed", "thu", "fri"])
campaign = st.sidebar.slider("Contacts During This Campaign", 1, 50, 1)
pdays = st.sidebar.slider(
    "Days Since Previous Campaign Contact", 0, 999, 999,
    help="999 = client was not contacted in a previous campaign.")
previous = st.sidebar.slider("Contacts in Previous Campaigns", 0, 10, 0)
poutcome = st.sidebar.selectbox("Previous Campaign Result", [
    "nonexistent", "failure", "success"])

# ── Sidebar: Economy ─────────────────────────────────────────────
st.sidebar.header("📈 Economic Indicators")
euribor3m = st.sidebar.slider(
    "Euribor 3-Month Rate (%)", 0.6, 5.1, 4.9, 0.1)
cons_price_idx = st.sidebar.slider(
    "Consumer Price Index", 92.0, 95.0, 93.9, 0.1)
emp_var_rate = st.sidebar.slider(
    "Employment Variation Rate (%)", -3.5, 1.5, 1.1, 0.1)
cons_conf_idx = st.sidebar.slider(
    "Consumer Confidence Index", -51.0, -26.0, -36.4, 0.1)
nr_employed = st.sidebar.slider(
    "Number of Employees (k)", 4960.0, 5230.0, 5191.0, 10.0)

# Engineered features (must match training pipeline)
CPI_BASE = 92.201  # same base as train_model.py
inflation_rate = ((cons_price_idx - CPI_BASE) / CPI_BASE) * 100
real_interest_rate = euribor3m - inflation_rate

# ── Predict ──────────────────────────────────────────────────────
st.divider()
if st.button("🔮 Predict", use_container_width=True):
    input_df = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan,
        "contact": contact, "month": month, "day_of_week": day_of_week,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "inflation_rate": inflation_rate,
        "real_interest_rate": real_interest_rate,
    }])

    prob = model.predict_proba(input_df)[0][1]
    st.subheader(f"Subscription Probability: {prob * 100:.2f}%")

    if prob > 0.5:
        st.success("✅ High-potential client — a marketing call is recommended.")
    else:
        st.info("ℹ️ Low-potential client — resource conservation is recommended.")

    with st.expander("📖 Economic Commentary"):
        real_rate = real_interest_rate
        st.markdown(
            f"**Fisher Equation:** Nominal {euribor3m:.2f}% − "
            f"Inflation {inflation_rate:.2f}% ≈ **Real Rate {real_rate:.2f}%**\n\n"
            + ("Positive real rate → deposits are attractive."
               if real_rate > 0 else
               "Negative real rate → deposits lose purchasing power.")
            + f"\n\n**Employment:** "
            + (f"Contracting ({emp_var_rate:.1f}%) — uncertainty reduces commitments."
               if emp_var_rate < 0 else
               f"Stable/growing ({emp_var_rate:.1f}%) — supports deposit decisions.")
        )

st.divider()
st.caption(
    "Model trained on Portuguese bank marketing data (2008–2013). "
    "Feature real_interest_rate = euribor3m − inflation_rate, "
    "where inflation_rate = ((CPI − 92.201) / 92.201) × 100 (Fisher equation)."
)
