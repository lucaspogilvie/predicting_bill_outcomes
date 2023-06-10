import pickle
import streamlit as st
import pandas as pd

df = pd.read_csv('data_by_state_2023.csv')

st.title('State Bill Prediction')

#Explaining the model
st.header('About the Model')
st.write(f'''This model uses the title of the bill, the majority party of the state's congressional \
 houses, the governor's party, and whether a state's houses and governor parties are alligned or not to \
 predict if the bill will pass or not. This model was \
 trained on 80,000 peices of legislation and tested on 27,000 bills from the years 2017 and 2018. \
 On the test set, the model was able to predict 75% of the bills that passed. This is impressive as \
 only 20% of state bills are enacted into law. When the model predicted the bill would not pass, it \
 was correct 92% of the time. When the model predicted the bill would pass though, it was only \
 correct 40% of the time. 

 A major limitation of this model is that this model's ability to interpret language is limited. For example, \
 it is difficult for the model to differentiate between phrases like "raise taxes" compared to "decrease taxes". \
''')

#file path to pickled model
with open('../04_Modeling/random_forest.pkl', 'rb') as pickle_in:
	model = pickle.load(pickle_in) 

#Text for user to put state bill title
st.header('Model Prediction')
bill_title = st.text_input('Copy and paste the state bill title.', max_chars = 1000)

#State where bill is being introduced
state = st.selectbox('What state is the bill being introduced in?', options = df)

#make dataframe with the wanted features from the state in question
#Dataframe is nice because it is needed for our pickled model
to_be_predicted = df[df['states'] == state][['gov_party','senate_party','house_party','state_party_control']]

#Add our title feature to the dataframe
to_be_predicted['title'] = bill_title

#make prediction
predicted_outcome = model.predict(to_be_predicted)

if predicted_outcome[0] == 0:
	outcome = 'fail'
else:
	outcome = 'pass'
if st.button('Predict Bill Outcome'):
    st.write(f'The model predicts that the bill will {outcome}')
