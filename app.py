import streamlit as st
import pandas as pd
from inference import predict_daily

st.set_page_config(page_title="Daily Stock Signals", layout="wide")
st.title("Daily Stock Signals")

# Load fixed daily data
df = pd.read_csv("data_today.csv")

# Run inference
result = predict_daily(df)

# Display
st.subheader("Top Signals Today")
st.dataframe(result.head(20), use_container_width=True)
