import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    tokens = []
    for word in text:
        if word.isalnum():
            tokens.append(word)

    tokens = [
        ps.stem(word)
        for word in tokens
        if word not in stopwords.words('english')
    ]

    return " ".join(tokens)

   

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        print("processing")
     

#Preprocess
    transformed_sms = transform_text(input_sms)


#Vectorize
    vector_input = tfidf.transform([transformed_sms])

#Predict
    result = model.predict(vector_input)[0]

#Display

    if result == 1:
        st.header("Spam Message Detected!")
    else:
        st.header("Not Spam, This message looks safe!")


st.markdown("---")
st.markdown(
    " **Kartik** | Build with ❤️ & ML(TF-IDF)"
)
