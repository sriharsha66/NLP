import numpy as np
import pandas as pd
from flask import Flask,request,jsonify
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
import main
from main import remove_noise

app = Flask(__name__)


@app.route("/tweet",methods=["POST"])
def predict():


    
    req = request.json
    print(req)
    try:
            
        custom_tweet = req["tweet_1"]
        custom_tweet_1 = req["tweet_2"]
        custom_tweet_2 = req["tweet_3"]

        custom_tokens = remove_noise(word_tokenize(custom_tweet))
        custom_tokens_1 = remove_noise(word_tokenize(custom_tweet_1))
        custom_tokens_2 = remove_noise(word_tokenize(custom_tweet_2))

        data1=[]
        data={
            "t1":main.classifier.classify(dict([token, True] for token in custom_tokens)),
            "t2":main.classifier.classify(dict([token, True] for token in custom_tokens_1)),
            "t3":main.classifier.classify(dict([token, True] for token in custom_tokens_2))
        }
        data1.append(data)
        return ({"respcode":"200","respdesc":"success","tweet":data1})
    except Exception as e:
        print(f'no data found:{e}')
        return ({"respcode":"404","respdesc":"failed","tweet":None})

if __name__ =="__main__":
    app.run(debug=True)

##################################


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
import pickle

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

stop_words = stopwords.words('english')

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
neutral_tweet_tokens = twitter_samples.tokenized('tweets.20150430-223406.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []
neutral_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
for tokens in neutral_tweet_tokens:
    neutral_cleaned_tokens_list.append(remove_noise(tokens,stop_words))

all_pos_words = get_all_words(positive_cleaned_tokens_list)
all_neutral_words=get_all_words(neutral_cleaned_tokens_list) # testing 
all_negative_words = get_all_words(negative_cleaned_tokens_list)

freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))

freq_dist_pos1 = FreqDist(all_negative_words)
print(freq_dist_pos1.most_common(10))

freq_dist_pos2 = FreqDist(all_neutral_words)
print(freq_dist_pos2.most_common(10))

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
neutral_tokens_for_model = get_tweets_for_model(neutral_cleaned_tokens_list) ##

positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

neutral_dataset = [(tweet_dict,"Neutral")
                        for tweet_dict in neutral_tokens_for_model]

dataset = positive_dataset + negative_dataset + neutral_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))


