import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import re 

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', "", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?:://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "True  News"

def manual_testing(news):
    testing_news = {"text": [news]}
    dt = load('DecisionTree.model')
    lr = load('LogisticRegression.model')
    rf = load('randomforest.model')
    
    vectorization = load('vectorizer.jb')

    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    predict_lr = lr.predict(new_xv_test)
    pred_dt = dt.predict(new_xv_test)
    pred_rf = rf.predict(new_xv_test)
    
    
    return "\n Logistic Regression  Prediction:{}  \n Decision Tree  Prediction:{}  \n RAndom Forest Classifier Prediction:{} \n ".format(output_label(predict_lr[0]), output_label(pred_dt[0]), output_label(pred_rf[0]))


while True:
    print("""The Menu are as following-:
          1. To Check News
          2. Press 0 to exit
""")
    choice=int(input("Enter Your Choice:"))
    if choice==0:
        print("Thank You For Using My Service ")
        break
    else:
        news = input("Enter the News For Checking : ")

    print(manual_testing(news))
    print(50*"*")
