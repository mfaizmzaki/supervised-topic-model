'''
Created on Jun 26, 2016

@author: mfaizmzaki
'''

from stot_model import stot_model
import pickle
import numpy as np
from gensim.corpora.dictionary import Dictionary

def main():
    
    articles_path = '/Users/mfaizmzaki/Dropbox/TextAnalysisPeacekeeping/texts_corrected/*.txt'
    stopword_path = '/Users/mfaizmzaki/Dropbox/TextAnalysisPeacekeeping/political-science/code/stopWordsFinal.txt'
    resultspath = '/Users/mfaizmzaki/Desktop/result/'
    location_path = '/Users/mfaizmzaki/Desktop/UCL_courses/mscproject/political-science/code/cities/locationVoterList2.txt'
    tot_topic_vectors_path = resultspath + 'time200msc_topic_vectors_beta0_1.csv'
    tot_topic_mixtures_path = resultspath + 'time200msc_topic_mixtures_beta0_1.csv'
    tot_topic_shapes_path = resultspath + 'time200msc_topic_shapes_beta0_1.csv'
    tot_pickle_path = resultspath + 'time200iter_beta0_1.pickle'
    coherence_pickle_path = resultspath + 'coherence.pickle'
    seed_file = resultspath + '/seedwords.txt'
    
    tot = stot_model()
 
    
    articles,date,vocab = tot.initDataset(articles_path, stopword_path, location_path)
    
    ##save variable for coherence measures
    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(article) for article in articles]
    
    coherence_pickle = open(coherence_pickle_path, 'wb')
    pickle.dump(dictionary, coherence_pickle)
    pickle.dump(corpus, coherence_pickle)
    coherence_pickle.close()
    
    #resume with modelling process
    tot.init_seedwords(seed_file, vocab)  
    param = tot.initParam(articles, date, vocab)
    theta,phi,psi = tot.TopicsOverTimeGibbsSampling(param)
    np.savetxt(tot_topic_vectors_path, phi, delimiter=',')
    np.savetxt(tot_topic_mixtures_path, theta, delimiter=',')
    np.savetxt(tot_topic_shapes_path, psi, delimiter=',')
    tot_pickle = open(tot_pickle_path, 'wb')
    pickle.dump(param, tot_pickle)
    tot_pickle.close()
    
if __name__ == "__main__":
    main()
        
        
        