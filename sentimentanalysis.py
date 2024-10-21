import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score
import re
import streamlit as sl
ps = PorterStemmer()
sns.set()





data = pd.read_csv("C:\\Users\\DELL\\Downloads\\Extended_Restaurant_Reviews.csv")
data




corpus = []

for i  in range(len(data)):
    s = re.sub('[^a-zA-Z]'," ",data ['Review' ][i])
    s = s.lower()
    s = s.split()
    s = [ word for word in s if word not in stopwords.words('english')]
    s = ' '.join(s)
    s = ps.stem(s)
    corpus.append(s)





cv=CountVectorizer()

X = cv.fit_transform(corpus).toarray()
y = data ['Liked'].values



X_train ,X_test, y_train ,y_test = train_test_split (X ,y ,test_size=0.25 ,random_state=42)





LR_model = LogisticRegression()

LR_model.fit(X_train , y_train)
y_pred = LR_model.predict(X_test)


acu_score = accuracy_score(y_test , y_pred)
print(acu_score)






sl.title('welcome to sentiment analysis app')
sl.markdown('---')



sl.markdown("""
            <style>
             .stDataFrame.st-emotion-cache-6nrhu6.e1w7nams0
             {
                display: none;
             }
            </style>
             """ , unsafe_allow_html=True)
form = sl.form('sentiment')
text = form.text_input('please enter your tweet\comment here :')
sl.markdown('---')


if form.form_submit_button('check'):
    text = re.sub('[^a-zA-Z]'," ",text)
    text = text.lower()
    text = text.split()
    text = [ word for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    text = ps.stem(text)
    prediction = LR_model.predict(cv.transform([text]).toarray())
    
    if prediction[0] == 1 :
        sl.write(f"the tweet\comment you entered is positive")
    else:
        sl.write(f"the tweet\comment you entered is negative")
        






























