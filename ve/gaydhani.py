import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#Code samples taken from:
#https://github.com/younggns/comparative-abusive-lang

#Orignal work says they used scikit for implementation, will use all details given and assume all other settings are the scikit defaults

def gaydhani(raw_data):

    name = 'Gaydhani et al (2018)'

    print('Preparing', name + '...')

    print('Pre-processing raw data...')

    pre_processed_data = [[preprocess_tweet(tweet), label] for tweet, label in raw_data]

    print('Building model...')

    # As per original paper, best results were with n-gram 1-3, TFIDF = L2 and C = 100, optimiser 'liblinear'
    classifier = Pipeline([('vect',
                            TfidfVectorizer(ngram_range=(1, 3), norm='l2')),
                           ('clf', LogisticRegression(C=100, solver='liblinear'))])

    print(name, 'prepared.')

    return name, classifier, pre_processed_data

def preprocess_tweet(tweet):

    RE_FLAGS = re.MULTILINE | re.DOTALL

    text = tweet.lower()

    final_text = text

    # Removes URLs
    final_text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", '', final_text, flags=RE_FLAGS)

    # Removes mentions
    final_text = re.sub(r"@\w+", '', final_text, flags=RE_FLAGS)

    # Removes Retweets
    retweet_prefix = 'rt :'

    if final_text.startswith(retweet_prefix):
        final_text = final_text[len(retweet_prefix):]

    # Although not strictly in the final paper, strip whitespace from text
    final_text = final_text.strip()

    # Removes whitespace and replaces single space
    final_text = re.sub(r"\s+", ' ', final_text, flags=RE_FLAGS)

    # Remove Stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = final_text.split(' ')
    final_text = ' '.join([w for w in word_tokens if not w in stop_words])

    # Stem with Porter Stemmer algorithm
    stemmer = PorterStemmer()
    final_text = stemmer.stem(final_text)

    return final_text