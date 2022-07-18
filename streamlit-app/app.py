import streamlit as st
import helper
import pickle
import cosine_similarity
model = pickle.load(open('model.pkl', 'rb'))
model_without_rules = pickle.load(open('model_without_rules_2.pkl', 'rb'))
st.header('Finding similar question pair -')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')
list_acc = ["0.798", "0.62"]
option = st.selectbox('Which method will you like to us', ('None','Cosine Simliarity', 'Random Forest Classifier', 'Random Forest Classifier without rules'))

if option == "Cosine Simliarity":
    # st.text("Accuracy :" + list_acc[0])
    pass
elif option == "Random Forest Classifier":
    st.text("Model Accuracy :" + list_acc[0])

elif option == "Random Forest Classifier without rules":
    st.text("Model Accuracy :" + list_acc[1])
else:
    pass

if st.button('Find'):
    if option == "Cosine Simliarity":
        query = cosine_similarity.fn_cos_sim(q1, q2)
        st.text("Percentage Similarity :" + str(query))
        if query>0.5:
            st.header('Similar Questions')
        else:
            st.header('Not Similar Question')

    elif option == "Random Forest Classifier":
        query = helper.query_point_creator(q1, q2)
        result = model.predict(query)[0]
        if result:
            st.header('Similar Questions')
        else:
            st.header('Not Similar Question')
    elif option == "Random Forest Classifier without rules":
        a = helper.query_point_creator(q1, q2)
        query = a[:,3:]
        result = model_without_rules.predict(query)[0]
        if result:
            st.header('Similar Questions')
        else:
            st.header('Not Similar Question')
        # query = helper.query_point_creator(q1, q2)
        # result = model.predict(query)[0]
        # if result:
        #     st.header('Not Similar Question')
        # else:
        #     st.header('Similar Questions')
