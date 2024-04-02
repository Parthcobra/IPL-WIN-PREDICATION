import streamlit as st
import pickle
import pandas as pd

pipe = pickle.load(open('IPL_Score.pkl', 'rb'))
dt = pickle.load(open('dataset_IPL_Score.pkl', 'rb'))

cities=['Hyderabad', 'Rajkot', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
       'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
       'Port Elizabeth', 'Durban', 'Centurion', 'East London',
       'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
       'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune',
       'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
       'Bengaluru']

st.title('IPL Win Predictor')

team1, team2 = st.columns(2)

with team1:
    bat_team = st.selectbox('Choose batting team', sorted(dt['batting_team'].unique()))
with team2:
    bowl_team = st.selectbox('Choose bowling team', sorted(dt['batting_team'].unique()))

city = st.selectbox('Choose the hosted city', sorted(cities))

target = st.number_input('Target')

bal, overs, wick = st.columns(3)

with bal:
    score = st.number_input('Score')
with overs:
    over = st.number_input('Overs thrown ')
with wick:
     wickets = st.number_input('Wickets Fallen ')

if st.button('Predict Winning Probability '):
    balls_left = 120 - (over * 6)
    runs_left = target-score
    wickets = 10-wickets
    crr = score/over
    rrr = (runs_left*6)/balls_left
    input = pd.DataFrame({'batting_team': [bat_team], 'bowling_team': [bowl_team], 'city': [city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'CRR': [crr], 'RRR': [rrr] })
    st.table(input)
    result = pipe.predict_proba(input)
    loss = result[0][0]
    win = result[0][1]
    st.subheader(bat_team + ": " + str(round(win*100)) + "%")
    st.subheader(bowl_team + ": " + str(round(loss*100)) + "%")