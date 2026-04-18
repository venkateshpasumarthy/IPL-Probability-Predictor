import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="IPL Win Predictor", layout="wide")

st.title("🏏 IPL Win Probability Predictor")

# -------- Input Section --------
col1, col2 = st.columns(2)

with col1:
    score = st.number_input("Current Score", min_value=0)
    overs = st.number_input("Overs Completed", min_value=1.0, max_value=20.0)
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)

with col2:
    target = st.number_input("Target Score", min_value=1)

# -------- Feature Calculation --------
runs_left = target - score
balls_left = 120 - int(overs * 6)
wickets_left = 10 - wickets

current_rr = score / overs if overs > 0 else 0
required_rr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# -------- Prediction --------
if st.button("Predict"):
    input_data = np.array([[runs_left, balls_left, wickets_left, current_rr, required_rr]])
    prob = model.predict_proba(input_data)[0]

    st.subheader("📊 Prediction Result")
    st.write(f"Winning Probability: **{round(prob[1]*100,2)}%**")
    st.write(f"Losing Probability: **{round(prob[0]*100,2)}%**")

    # -------- Explanation --------
    st.subheader("🧠 Match Insight")

    if required_rr > 10:
        st.warning("High pressure: Required Run Rate is high")
    elif wickets_left < 3:
        st.warning("Low stability: Very few wickets left")
    else:
        st.success("Balanced match situation")

    # -------- Graph --------
    st.subheader("📈 Win Probability Progression")

    probs = []
    overs_list = list(range(1, int(overs)+1))

    for o in overs_list:
        temp_balls = 120 - o*6
        temp_rrr = (runs_left * 6) / temp_balls if temp_balls > 0 else 0

        temp_input = np.array([[runs_left, temp_balls, wickets_left, current_rr, temp_rrr]])
        p = model.predict_proba(temp_input)[0][1]
        probs.append(p)

    fig, ax = plt.subplots()
    ax.plot(overs_list, probs)
    ax.set_xlabel("Overs")
    ax.set_ylabel("Win Probability")
    ax.set_title("Match Progression")

    st.pyplot(fig)
