import re
from nltk import stem
import json

RE_FLAGS = re.MULTILINE | re.DOTALL

import re
import enchant
import splitter
from sklearn import metrics

def mondal(raw_data, target_label, fallback_label, all_original_labels):

    name = 'Mondal (2018)'

    print('Running', name + '...')

    processed_tweets = []

    stemmer = stem.PorterStemmer()

    d = enchant.Dict('en_UK')
    dus = enchant.Dict('en_US')

    print('Preparing token lists...')

    openers, intensity_terms, user_intent_terms, target_exclusion_words, hate_targets = prepare_tokens(stemmer)

    print('Pre-processing tweets...')

    for tweet, original_label in raw_data:

        processed_tweet = preprocess_tweet(tweet, d, dus, stemmer)

        mapped_label = original_label

        if not mapped_label == target_label:
            mapped_label = fallback_label

        contains_very_offensive_slur = False
        contains_any_slur = False

        for hate_target, is_very_offensive in hate_targets:
            if contains_any_slur == True and contains_very_offensive_slur == True:
                break

            if len(hate_target) == 1:
                if hate_target[0] in processed_tweet:
                    contains_any_slur = True
                    contains_very_offensive_slur  = is_very_offensive
            else:
                for i in range(0, len(processed_tweet) - len(hate_target)):
                    if processed_tweet[i:i + len(hate_target)] == hate_target:
                        contains_any_slur = True
                        contains_very_offensive_slur = is_very_offensive

        processed_tweets.append((tweet, processed_tweet, mapped_label, original_label, contains_very_offensive_slur, contains_any_slur))


    total_any_slur = 0
    total_offensive_slur = 0

    print('texts containing offensive hatebase terms')
    print('label,frequency,ratio')

    for label in all_original_labels:

        label_count = len([_ for _, _, _, original_label, _, _ in processed_tweets if original_label == label])

        offensive_slur = len([original_label for _, _, _, original_label, contains_very_offensive_slur, _ in processed_tweets if original_label == label and contains_very_offensive_slur == True])

        total_offensive_slur = total_offensive_slur + offensive_slur

        print('{' + label + '},' + str(offensive_slur) + ',' + percent(offensive_slur/label_count * 100))

    print('total,' + str(total_offensive_slur) + ',' + percent(total_offensive_slur/len(processed_tweets) * 100))

    print('texts containing any hatebase terms')
    print('label,frequency,ratio,growth')

    for label in all_original_labels:

        label_count = len([_ for _, _, _, original_label, _, _ in processed_tweets if original_label == label])

        any_slur = len([original_label for _, _, _, original_label, _, contains_any_slur in processed_tweets if original_label == label and contains_any_slur == True])
        offensive_slur = len([original_label for _, _, _, original_label, contains_very_offensive_slur, _ in processed_tweets if original_label == label and contains_very_offensive_slur == True])

        total_any_slur = total_any_slur + any_slur

        print('{' + label + '},' + str(any_slur) + ',' + percent(any_slur / label_count * 100) + ',' + percent(((any_slur/offensive_slur) - 1) * 100))

    print('total,' + str(total_any_slur) + ',' + percent(total_any_slur/len(processed_tweets) * 100) + ',' + percent(((total_any_slur/total_offensive_slur) - 1) * 100))

    print('Running searching...')

    #Template 1: I <intensity> <user intent> <any word> people
    #Template 2: I <intensity> <user intent> <hate target>

    results = []

    for raw_tweet, processed_tweet, mapped_label, original_label, cvos, cs in processed_tweets:

        final_label = fallback_label

        for opener in openers:

            if not opener in processed_tweet:
                continue

            opener_indicies = [i for i, x in enumerate(processed_tweet) if x == opener]

            for opener_index in opener_indicies:

                sentence = processed_tweet[opener_index + len(opener):]

                is_searching_intensity_terms = True

                while is_searching_intensity_terms == True:

                    is_searching_intensity_terms = False

                    for intensity_term in intensity_terms:
                        if sentence[:len(intensity_term)] == intensity_term:
                            sentence = sentence[len(intensity_term):]
                            is_searching_intensity_terms = True
                            break

                user_intent_found = False

                is_searching_user_intent_terms = True

                while is_searching_user_intent_terms == True:

                    is_searching_user_intent_terms = False

                    for user_intent in user_intent_terms:
                        if sentence[:len(user_intent)] == user_intent:
                            sentence = sentence[len(user_intent):]
                            user_intent_found = True
                            is_searching_user_intent_terms = True
                            break

                if not user_intent_found:
                    continue

                is_searching_exclusion_words = True

                while is_searching_exclusion_words == True:

                    is_searching_exclusion_words = False

                    for exclusion_word in target_exclusion_words:
                        if sentence[:len(exclusion_word)] == exclusion_word:
                            sentence = sentence[len(exclusion_word):]
                            is_searching_exclusion_words = True
                            break

                if len(sentence) > 1 and sentence[1] == stemmer.stem('people'):
                    #print('Template 1 detected hateful:', (raw_tweet, processed_tweet, label))
                    final_label = target_label
                else:
                    for hate_target, is_very_offensive in hate_targets:
                        if is_very_offensive == True and sentence[:len(hate_target)] == hate_target:
                            #print('Template 2 detected hateful:', (raw_tweet, processed_tweet, label))
                            final_label = target_label
                            break

        results.append((raw_tweet, processed_tweet, final_label, original_label, cvos, cs))

    truth = [mapped_label for raw, tweet, mapped_label, original_label, cvos, cs in processed_tweets]

    predictions = [final_label for raw, tweet, final_label, original_label, cvos, cs in results]

    return name, predictions, truth

def prepare_tokens(stemmer):

    openers = ['i']
    #small alteration for 'cannot' to 'can not'
    hate_synonyms = ['hate', 'do not like', 'abhor', 'despise', 'detest', 'loathe', 'scorn', 'shun',
                     'abominate', 'anathematize', 'contemn', 'curse', 'deprecate', 'deride',
                     'disapprove', 'disdain', 'disfavor', 'disparage', 'execrate', 'nauseate',
                     'spurn', 'am allergic to', 'am disgusted with', 'am hostile to', 'am loath',
                     'am reluctant', 'am repelled by', 'am sick of', 'bear a grudge against',
                     'can not stand', 'down on', 'feel malice to', 'have an aversion to',
                     'have enough of', 'have no use for', 'look down on', 'do not care for',
                     'object to', 'recoil from', 'shudder at', 'spit upon']

    stemmed_hate_synonyms = []

    for symbol in hate_synonyms:
        stemmed = stemmer.stem(symbol)
        stemmed_hate_synonyms.append(stemmed.split(' '))

    intensity_tokens = ['absolute', 'absolutely', 'actually', 'already', 'also', 'always', 'bloody', 'completely',
                        'definitely', 'do', 'especially', 'extremely', 'f*cking', 'fckin', 'fkn', 'fr', 'freakin',
                        'freaking', 'fucken', 'fuckin', 'fucking', 'fuckn', 'generally', 'genuinely', 'honestly',
                        'honesty', 'jus', 'just', 'kinda', 'legitimately', 'literally', 'naturally', 'normally', 'now',
                        'officially', 'only', 'passively', 'personally', 'proper', 'really', 'realy', 'rlly', 'rly',
                        'secretly', 'seriously', 'simply', 'sincerely', 'so', 'sometimes', 'sorta', 'srsly', 'still',
                        'strongly', 'totally', 'truly', 'usually']

    stemmed_intensity_tokens = []

    for symbol in intensity_tokens:
        stemmed = stemmer.stem(symbol)
        stemmed_intensity_tokens.append(stemmed.split(' '))

    exclusion_words = ['about', 'all', 'any', 'asking', 'disappointing', 'everyone', 'following', 'for', 'having',
                       'hearing', 'how', 'hurting', 'is', 'it', 'letting', 'liking', 'many', 'meeting', 'more',
                       'most', 'my', 'myself', 'on', 'other', 'seeing', 'sexting', 'some', 'telling',
                       'texting', 'that', 'the', 'them', 'these', 'this', 'those', 'watching', 'wen',
                       'what', 'when', 'when', 'whenever', 'why', 'with', 'you']

    stemmed_exclusion_words = []

    for symbol in exclusion_words:
        stemmed = stemmer.stem(symbol)
        stemmed_exclusion_words.append(stemmed.split(' '))

    stemmed_hate_target_terms = []

    hate_target_terms = generate_hatebase_terms()

    for symbol, is_very_offensive in hate_target_terms:
        stemmed = stemmer.stem(symbol.replace("'",''))
        stemmed_hate_target_terms.append((stemmed.split(' '), is_very_offensive))

    stemmed_intensity_tokens.sort(reverse=True, key=len)
    stemmed_hate_synonyms.sort(reverse=True, key=len)
    stemmed_exclusion_words.sort(reverse=True, key=len)

    return openers, stemmed_intensity_tokens, stemmed_hate_synonyms, stemmed_exclusion_words, stemmed_hate_target_terms

def preprocess_tweet(tweet, d, dus, stemmer):

    final_text = tweet.lower()

    final_text = final_text.replace("'", '')
    final_text = final_text.replace('’', '')
    final_text = final_text.replace('"', ' ')

    final_text = final_text.replace('&amp;', ' ')
    final_text = final_text.replace('&lt;', ' ')
    final_text = final_text.replace('&gt;', ' ')

    final_text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", '', final_text, flags=RE_FLAGS)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    final_text = emoji_pattern.sub(r' ', final_text)

    # Borrowed from Zhang code
    final_text = re.sub('&#[0-9]{4,6};', ' ', final_text, flags=RE_FLAGS)

    final_text = final_text.replace('.', ' ')
    final_text = final_text.replace('?', ' ')
    final_text = final_text.replace(',', ' ')
    final_text = final_text.replace('!', ' ')
    final_text = final_text.replace('‼', ' ')
    final_text = final_text.replace('|', ' ')
    final_text = final_text.replace('-', ' ')
    final_text = final_text.replace('+', ' ')
    final_text = final_text.replace('…', ' ')
    final_text = final_text.replace('—', ' ')
    final_text = final_text.replace(':', ' ')
    final_text = final_text.replace('/', ' ')
    final_text = final_text.replace('(', ' ')
    final_text = final_text.replace(')', ' ')
    final_text = final_text.replace('[', ' ')
    final_text = final_text.replace(']', ' ')
    final_text = final_text.replace('{', ' ')
    final_text = final_text.replace('}', ' ')
    final_text = final_text.replace('\u200d', ' ')

    final_text = re.sub(r"\s+", ' ', final_text, flags=RE_FLAGS)

    hashtags = re.findall('#[\w\-]+', final_text)

    for tag in hashtags:
        cleantag = tag[1:].lower()
        if d.check(cleantag) or dus.check(cleantag):
            final_text = re.sub(tag, cleantag, final_text)
        else:
            split_hashtag_words = []
            for word in splitter.split(cleantag.lower(), 'en_US'):
                split_hashtag_words.append(word.lower())

            if len(split_hashtag_words) == 0:
                final_text = re.sub(tag, cleantag, final_text)
            else:
                final_text = re.sub(tag, ' '.join(split_hashtag_words), final_text)

    final_text = re.sub(r"\s+", ' ', final_text, flags=RE_FLAGS)

    tokenised = final_text.split(' ')

    final_tokens = []

    for token in tokenised:

        if token == 'im':
            final_tokens.append('i')
            final_tokens.append('am')
        elif token == 'u':
            final_tokens.append('you')
        elif token == '@':
            final_tokens.append('at')
        elif token == '$':
            pass
        elif '$' in token:
            final_tokens.append(token.replace('$', 's'))
        elif token =='ill':
            final_tokens.append('i')
        elif token == 'dont':
            final_tokens.append('do')
            final_tokens.append('not')
        elif token == 'cant' or token == 'cannot':
            final_tokens.append('can')
            final_tokens.append('not')
        elif token == 'and' or token == 'or':
            pass
        else:
            final_tokens.append(token)

    final_text = ' '.join(final_tokens)

    final_text = final_text.replace('#', ' ')

    final_text = re.sub(r"\s+", ' ', final_text, flags=RE_FLAGS)

    final_text = stemmer.stem(final_text)

    final_text = re.sub(r"\s+", ' ', final_text, flags=RE_FLAGS)

    final_text = final_text.strip()

    return final_text.split(' ')

def percent(value, dp = 2):

    parts = str(round(value, dp)).split('.')

    int = parts[0]
    dec = '0'

    if len(parts) == 2:
        dec = parts[1]

    return int  + '.' + dec.ljust(dp, '0')

def generate_hatebase_terms():

    print('Loading vocabulary from Hatebase...')

    all_words = []

    filtered_words = []

    all_results = []

    for i in range(1,17):

        filename = 'hatebase_scrape/page_' + str(i) + '.json'

        #print('Loading Hatebase results file:', filename)

        with open(filename, 'r') as file:
            contents = json.load(file)
            results = contents['result']
            all_results.extend(results)

    print('Number of raw Hatebase terms:', len(all_results))
    print('Number of raw Hatebase terms without average_offensiveness:', len([r for r in all_results if r['average_offensiveness'] is None]))

    for result in all_results:

        singular_term = result['term'].lower()
        all_words.append(singular_term)

        plural = result['plural_of']

        variation = result['variant_of']

        if not variation is None:
            variation = variation.lower()
            all_words.append(variation)

        if not plural is None:
            plural = plural.lower()
            all_words.append(plural)

        offensiveness = result['average_offensiveness']

        if not offensiveness is None and offensiveness > 50:
            filtered_words.append(singular_term)

            if not variation is None:
                filtered_words.append(variation)

            if not plural is None:
                filtered_words.append(plural)

    all_words = list(set(all_words))

    print('Number of Hatebase terms:', len(all_words))

    filtered_words = list(set(filtered_words))

    print('Number of Hatebase terms with average_offensiveness > 50:', len(filtered_words))

    #return all_words

    final_words = []

    for word in all_words:

        is_very_offensive = word in filtered_words

        final_words.append((word, is_very_offensive))

    return final_words