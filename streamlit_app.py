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
  df3 = df1[df1['Age Class'] == 'Adult']
  len(df3)
  st.write("I have filtered out the data to be only adults to simplify the processes, and as you can see there are only 510 out of a thousand in the dataset which are adults")
  df3.head()
  
  
  fig, ax = plt.subplots(figsize=(10, 10))
  plt.title("Length to weight ratio")
  sns.scatterplot(data = df3, x = 'Observed Length (m)', y = 'Observed Weight (kg)',hue = 'Common Name')
  plt.legend()
  st.pyplot(fig)

with st.sidebar:
  st.header("Input Features")
  observed_length = st.slider('Observed length (m)', 0.14,6.12)
  observed_weight = st.slider('Observed weight (kg)', 4.4, 1139.7)
