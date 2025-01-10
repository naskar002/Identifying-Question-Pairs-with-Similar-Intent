import streamlit as st
import supporter
import pickle

model = pickle.load(open('model.pkl','rb'))

st.header("Duplicate Question Pairs finder")

q1 = st.text_area('Enter question 1')
q2 = st.text_area('Enter question 2')

if st.button('Find'):
    query = supporter.query_pt_creator(q1,q2)
    result = model.predict(query)[0]
    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')