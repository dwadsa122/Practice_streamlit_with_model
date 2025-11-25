import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Title setting
st.title('Practice Tool (Streamlit with machine learning')


#Display the Data
with st.expander("Data"):
  st.write('**Raw Data**')

  # Load the Data and drop unnecessary columns
  df = pd.read_csv("crocodile_dataset.csv")
  df1 = df.drop(columns = df[['Observation ID','Scientific Name','Genus','Genus','Conservation Status', 'Observer Name', 'Notes','Family','Date of Observation']])

  # Create the raw X and Y Values
  st.write("It is important to note that there are various classifications in the common name column, lets remove some columns that seem to be unnecessary, and we'll only keep those that we need.")
  st.write("Below shall be the X values with all of its features")
  X = df.drop(columns = df[['Observation ID','Scientific Name','Genus','Genus','Conservation Status', 'Observer Name', 'Notes','Family','Date of Observation', 'Common Name']])
  X
  
  st.write("And here shall be our class target")
  y = df['Common Name']
  y

# Setup visualizations
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

# Setup Input Features
with st.sidebar:
  st.header("Input Features")
  observed_length = st.slider('Observed length (m)', 0.14,6.12)
  observed_weight = st.slider('Observed weight (kg)', 4.4, 1139.7)
  Age_class = st.selectbox('Age Class', ('Adult', 'SubAdult', 'Juvenile', 'Hatchling'))
  Sex = st.selectbox('Sex', ('Male','Female','Unknown'))
  Region = st.selectbox('Country/Region', ('Australia', 'Belize', 'Cambodia', 'Cameroon',
       'Central African Republic', 'Chad', 'Colombia', 'Congo (DRC)',
       'Congo Basin Countries', 'Costa Rica', 'Cuba', "Côte d'Ivoire",
       'Egypt', 'Gabon', 'Ghana', 'Guatemala', 'Guinea', 'India',
       'Indonesia', 'Indonesia (Borneo)', 'Indonesia (Papua)',
       'Iran (historic)', 'Kenya', 'Laos', 'Liberia', 'Malaysia',
       'Malaysia (Borneo)', 'Mali', 'Mauritania', 'Mexico', 'Nepal',
       'Niger', 'Nigeria', 'Pakistan', 'Papua New Guinea', 'Philippines',
       'Senegal', 'Sierra Leone', 'South Africa', 'Sri Lanka', 'Sudan',
       'Tanzania', 'Thailand', 'USA (Florida)', 'Uganda', 'Venezuela',
       'Vietnam'))
  Habitat = st.selectbox('Describe its Habitat', ('Billabongs', 'Brackish Rivers', 'Coastal Lagoons',
       'Coastal Wetlands', 'Estuaries', 'Estuarine Systems',
       'Flooded Savannas', 'Forest Rivers', 'Forest Swamps',
       'Freshwater Marshes', 'Freshwater Rivers', 'Freshwater Wetlands',
       'Gorges', 'Lagoons', 'Lakes', 'Large Rivers', 'Mangroves',
       'Marshes', 'Oases', 'Oxbow Lakes', 'Ponds', 'Reservoirs', 'Rivers',
       'Shaded Forest Rivers', 'Slow Rivers', 'Slow Streams',
       'Small Streams', 'Swamps', 'Tidal Rivers'))

# Create dataframe from input features
  data = {'Observed Length (m)': observed_length,
          'Observed Weight (kg)': observed_weight,
          'Age Class': Age_class,
          'Sex': Sex,
          'Country/Region': Region,
          'Habitat Type': Habitat}
  input_df = pd.DataFrame(data, index = [0])
  input_croc = pd.concat([input_df.head(1), X], axis = 0 , ignore_index=True)
with st.expander('Input Features Dataframe'):
  st.write("Here are the your input features")
  input_df
  st.write("Concatenated with the dataframe!")
  input_croc


st.subheader("Modeling")
st.write("We shall use CatBoosting for our modeling to compensate with the multiclass target that we have")
st.write("I have already made the model in a jupyter notebook and saved it here in the repository")
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("crocodile_model.cbm")

input_row = input_croc[0:]

prediction = model.predict(input_row)
pred_proba = model.predict_proba(input_row)[0]
classes = model.classes_

df_prob = pd.DataFrame({"Classes": classes, "Probability": pred_proba}).set_index("Classes")
st.dataframe(df_prob)





 
  
  
       
      

                     
