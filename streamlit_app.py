import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Practice Tool (Streamlit with machine learning')


with st.expander("Data"):
  st.write('**Raw Data**')
  
  df = pd.read_csv("crocodile_dataset.csv")
  df1 = df.drop(columns = df[['Observation ID','Scientific Name','Genus','Genus','Conservation Status', 'Observer Name', 'Notes','Family','Date of Observation']])
  
  st.write("It is important to note that there are various classifications in the common name column, lets remove some columns that seem to be unnecessary, and we'll only keep those that we need.")
  st.write("Below shall be the X values with all of its features")
  X = df.drop(columns = df[['Observation ID','Scientific Name','Genus','Genus','Conservation Status', 'Observer Name', 'Notes','Family','Date of Observation', 'Common Name']])
  X
  
  st.write("And here shall be our class target")
  y = df['Common Name']
  y

with st.expander("Data Visualizations"):
  plt.figure(figsize = (10,10))
  sns.scatterplot(data = df2, x = 'Observed Length (m)', y = 'Observed Weight (kg)',hue = 'Common Name')
  plt.legend()
