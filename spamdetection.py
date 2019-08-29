# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:36:49 2019

@author: Pushkar
"""



import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df=pd.read_csv("youtubecommentsdataset.csv")

df_data=df[['CONTENT','CLASS']]


df_x=df_data['CONTENT']
df_y=df_data['CLASS']

#converting test set to count vectors
cv=CountVectorizer()
x=cv.fit_transform(df_x)

x.toarray()

#splitting into train and test
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix

x_train,x_test,y_train,y_test=train_test_split(x,df_y,test_size=0.33,random_state=42)

#naive bayes
#from sklearn.naive_bayes import MultinomialNB
#clf=MultinomialNB()
#clf.fit(x_train,y_train)
#res=clf.score(x_test,y_test)
#nbpred=clf.predict(x_test)

#print("percentage of accurate result is ",res*100,"%")

#logistic regression
#from sklearn.linear_model import LogisticRegression

#linclf = LogisticRegression(solver='liblinear', penalty='l1')
#linclf.fit(x_train,y_train)
#lrpred = linclf.predict(x_test)
#acc=accuracy_score(y_test,lrpred)


#random forest Classification

from sklearn.ensemble import RandomForestClassifier

#clf1=RandomForestClassifier(criterion='entropy',min_samples_split=4,max_features='log2')
clf1=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features=8, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf1.fit(x_train,y_train)
rfpred=clf1.predict(x_test)

#the different scores for the model
print(accuracy_score(y_test,rfpred))
print(roc_auc_score(y_test,rfpred))
print(confusion_matrix(y_test,rfpred))



from flask import Flask, request, render_template

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

#rendering the html page for data input
@app.route('/')
def my_form():
    return render_template('my-form.html')


#taking the data and predicting whether its spam or not
@app.route('/', methods=['POST'])
def my_form_post():
    comment = request.form['comment']
    comment=[comment]
    val=cv.transform(comment).toarray()
    prediction=clf1.predict(val)
    if(prediction==0):
        #pred="Spam"
        msg = "not spam"
    else:
        msg = "spam"
#it renders html page which shows the result
    return render_template('result.html',pred=msg)

if __name__ == '__main__':
    
    app.run(port=5000,debug=True, use_reloader=False)
