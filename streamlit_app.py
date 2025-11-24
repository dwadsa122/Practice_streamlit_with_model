import streamlit as st

st.title('Practice Tool (Streamlit with machine learning')

st.write('This is my attempt to create a streamlit app with a machine learning model inside.')

with st.expander("Data"):
  st.write('**Raw Data**')
  df = pd.read_csv("crocodile_dataset.csv")
  df
