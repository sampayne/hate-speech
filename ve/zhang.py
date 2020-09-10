from zhang_classifier_dnn import gridsearch
import gensim.downloader as api
import datetime
import gensim
def zhang(raw_data):

    name = 'Zhang et al (2018)'

    print('Preparing', name + '...')

    print('Loading embeddings...')

    emb_models = []
    emb_models.append(gensim.models.KeyedVectors.load_word2vec_format('Vectors/word2vec-google-news-300', binary=True))

    classifier, tweets = gridsearch(raw_data=raw_data,
               labels=[], #labels=labels, #params["input"],
               model_descriptor='dropout=0.2,conv1d=100-4,maxpooling1d=4,gru=100-True,gmaxpooling1d,dense=3-softmax-none-0.01_0.01',#params["model_desc"],  # model descriptor
               word_norm_option=0, #params["word_norm"],  # 0-stemming, 1-lemma, other-do nothing
               randomize_strategy=0, #params["oov_random"],  # 0-ignore oov; 1-random init by uniform dist; 2-random from embedding
               pretrained_embedding_models=emb_models,
               expected_embedding_dim=300,
               word_dist_features_file=None)#wdist_file,
               #)use_mixed_data)

    final_tweets = []

    for i in range(0, len(raw_data)):
        final_tweets.append([tweets[i], raw_data[i][1]])

    print(name, 'prepared.')

    return name, classifier, final_tweets