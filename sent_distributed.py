import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
from nltk.stem.porter import  *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words('english'))

#Base category for warnings about deprecated features (ignored by default).
warnings.filterwarnings("ignore",category=DeprecationWarning)

#the %matplotlib inline will make your plot outputs appear and be stored within the notebook.

%matplotlib inline

train =pd.read_csv('/home/rahul/Desktop/Till August/Project_sem_5/train_E6oV3lV.csv')
test = pd.read_csv('/home/rahul/Desktop/Till August/Project_sem_5/test_tweets_anuFYb8.csv')
print (train.head())
print (test.head())




combin=train.append(test,ignore_index=True)
#print (combin['tweet'])
line=[]
#removing twitter handles (@user) and making a new column
#combin['tidy_tweet'] = np.vectorize(remove_pattern)(combin['tweet'], "@[\w]*")
# remove special characters, numbers, punctuations

combin['tidy_tweet'] = combin['tweet'].str.replace("[^a-zA-Z#]", " ") 

#Removing Short Words

combin['tidy_tweet'] = combin['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#print(combin.head())


#tokenization
tokenized_tweet = combin['tidy_tweet'].apply(lambda x: x.split())


#print (tokenized_tweet)
#Stemming is done in this part
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x])
#print(tokenized_tweet)


#join all tokenized and stemmed tokens together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=' '.join(tokenized_tweet[i])
combin['tidy_tweet'] = tokenized_tweet
print(combin['tidy_tweet'].head())




def hashtag_extract(x):
    hashtag=[]
    for i in x:
        ht =re.findall(r"(\w+)", i)
        hashtag.append(ht)
        
    return hashtag

# extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(combin['tidy_tweet'][combin['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combin['tidy_tweet'][combin['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

#Plotting non racist hastag
a = nltk.FreqDist(HT_regular)
a.plot(10)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()



b = nltk.FreqDist(HT_negative)

b.plot(10)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
vectors=vectorizer.fit_transform(combin['tidy_tweet'])
print(vectors)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


train_vec = vectors[:31962,:]
test_vec = vectors[31962:,:]
#print(test_vec)

#Now splitting data into training and testing set

xtrain_vec,xtest_vec ,ytrain,ytest = train_test_split(train_vec,train['label'],random_state=10,test_size=0.2)

lreg = LogisticRegression()
#training the model
lreg.fit(xtrain_vec,ytrain)

#prediction
#prediction on testing data
predict = lreg.predict_proba(xtest_vec) 
#print(predict)
prediction_int = predict[:,1] >=0.3 #if prediction greater than or equal to 0.3 then 1 else 0
#print(prediction_int[:10])
prediction_int = prediction_int.astype(np.int)
print(prediction_int[:10])
#calculating f1_score on test and prediction data
from sklearn.metrics import classification_report
print(classification_report(ytest, prediction_int))

f1_score(ytest,prediction_int)


