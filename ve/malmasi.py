import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

def malmasi(raw_data):

    name = 'Malmasi & Zampieri (2018)'

    print('Preparing', name  + '...')

    print('Pre-processing raw data...')

    pre_processed_data = [[preprocess_tweet(tweet), label] for tweet, label in raw_data]

    print('Building model...')

    classifier = Pipeline([('vect',
                            CountVectorizer(ngram_range=(2, 4), analyzer='char')),
                           ('clf', LinearSVC())])

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

    # https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    # Removes emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    final_text = emoji_pattern.sub(r'', final_text)

    # Borrowed from Zhang code
    final_text = re.sub('&#[0-9]{4,6};', '', final_text)

    # Although not strictly in the final paper, strip whitespace from text
    final_text = final_text.strip()

    return final_text