'''
Created on Jun 25, 2016

@author: mfaizmzaki
'''

import timing
import copy
import re
import glob
import scipy.special, random
import dateparser, datetime
import unicodedata
import numpy as np
from string import punctuation
from nltk.stem.snowball import FrenchStemmer
from nltk.stem.snowball import EnglishStemmer



class stot_model:
    
    def __init__(self):
        self.param = {}
        self.param['topiclabel'] = []
        self.param['topic_id'] = {}
        self.param['topic_label'] = {}
        
        
    #method to remove all the punctuations
    def strip_punctuation(self, s):
        return [''.join(c for c in x if c not in punctuation) for x in s]
    
    def getDate(self, doc_path):
        with open(doc_path) as doc:
            for line in doc:
                if line.startswith('Date:'):
                    listofDates = re.findall("[0-9]*\s?[a-zA-z]+\s?[0-9]{0,4}|aujourd'hui|hier|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}", line)
                    return listofDates[1] #return the first date
                    break
                
    def convertDateToSeconds(self, dateString):
        
        dateFormatted = dateparser.parse(dateString, settings={'PREFER_DAY_OF_MONTH': 'last'}).date()
        dateSeconds = (dateFormatted - datetime.date(1970,1,1)).total_seconds()
        
        return dateSeconds
    
    #method to remove the stopwords from the articles   
    def remove_stopwords(self, doc, stopwords_path):
        stopWords = []
        
        with open(stopwords_path) as stopwords:
            for stopword in stopwords:
                lines = unicode(stopword, 'utf-8', errors='replace')
                lines = unicodedata.normalize('NFKD', lines).encode('ascii', 'ignore')
                stopWords = stopWords + lines.split()
        
        stop_words = set(stopWords)
            
        for sw in stop_words.intersection(doc):
            while sw in doc:
                doc.remove(sw)
        #return a list        
        return doc
    
    def remove_location(self,doc,location_path):
        stopWords = []
        
        with open(location_path) as stopwords:
            for stopword in stopwords:
                lines = unicode(stopword.lower(), 'utf-8', errors='replace')
                lines = unicodedata.normalize('NFKD', lines).encode('ascii', 'ignore')
                stopWords = stopWords + lines.split()
        
        stop_words = set(stopWords)
            
        for sw in stop_words.intersection(doc):
            while sw in doc:
                doc.remove(sw)
        #return a list        
        return doc
                
    #method to stem the article using python's snowball stemmer   
    def stemArticle(self, doc):
        stemmer_fr = FrenchStemmer()
        stemmer_en = EnglishStemmer()
        
        stemmedArticle = [str(stemmer_fr.stem(w)) for w in doc]
        stemmedArticle = [str(stemmer_en.stem(w)) for w in stemmedArticle]   
        
        return stemmedArticle
    
    def addtopic(self, topiclabel):
        if topiclabel not in self.param['topic_label']:
            topic_id = len(self.param['topic_id'])
            self.param['topic_id'][topiclabel] = topic_id
            self.param['topic_label'][topic_id] = topiclabel
        
        return self.param['topic_id'][topiclabel]
    

        
    #initializing the dataset
    def initDataset(self, articles_path, stopword_path, location_path):
        
        articles = []
        date = []
        vocab = set()
        
        
        files = glob.iglob(articles_path)
        
        while True:
            try:
                doc = files.next()
                with open(doc) as article:
                    i = 0
                    self.param['topiclabel'].append([])
                    words = []
                    articleDate = self.getDate(doc)
                    articleDate = self.convertDateToSeconds(articleDate)
                    
                    date.append(articleDate)
                    
                    for line in article:
                        if line.startswith('Topic_label: '):
                            row = line.split(' ')
                            topic_label = row[1].rstrip()
                            topic_int = self.addtopic(topic_label)
                            self.param['topiclabel'][i].append(topic_int)
                            continue
                        
                        if line.startswith('Date: '):
                            continue
                        #line = self.strip_punctuation(line)
                        lines = unicode(line, 'utf-8', errors='replace')
                        lines = unicodedata.normalize('NFKD', lines).encode('ascii', 'ignore')
                        word = filter(None, re.split("[\- ']+", lines))
                        word = self.strip_punctuation(word)
                        word = [w.strip() for w in word if len(w) > 2]
                        words.extend(word)
                        
                i = i + 1        
                #convert word list into lower case        
                words = [w.lower() for w in words] 
                
                #remove stopwords
                words = self.remove_stopwords(words, stopword_path)
                words = self.remove_location(words, location_path)
                words = [w if w not in ['gbagbo','ouattara','laurent','republicain','fpi','nguessan','alassane'] else w.replace(w,'party') for w in words]
                words = [w if w not in ['frci','fanci','lieutenantcolonel','force republicain'] else w.replace(w,'military') for w in words]
                words = [w if w not in ['casqu','casques','bleu','bleus','didier','hote','nigerien','ghaneen','marocain'  \
                                        ,'pakistan','pakistanais','jordanien','togol','togolais','bangladesh','bangladeshian' \
                                        ,'benin', 'beninais', 'burkina', 'faso', 'burkina faso', 'burkinabe', 'senegal', 'senegalais'] \
                        else w.replace(w,'pkomil') for w in words]
                words = [w if w not in ['unpol'] else w.replace(w,'pkopolice') for w in words]
                words = [w if w not in ['onuci','unoci','choi','nation','uni','unies','yj','munzu','aichatou','mindaoudou','koenders','berd'] \
                         else w.replace(w,'pko') for w in words]
                
                noDigits_words = [word for word in words if not word.isdigit()]
                
                stem_words = self.stemArticle(noDigits_words)
                
                articles.append(stem_words)
                vocab.update(set(stem_words))
                print len(articles)
                
            except StopIteration:
                break   
    
        vocab = list(vocab)
        date_sortedIndex = np.argsort(date)
        date.sort()
        articles = [articles[i] for i in date_sortedIndex]
        first_date = date[0]
        last_date = date[len(date) - 1]
        self.param['date'] = date
        print date[0]
        print date[len(date) - 1]
        date = [1.0*(d-first_date)/(last_date-first_date) for d in date]
        assert len(articles) == len(date)
        self.param['topicseed'] = [[] for _ in range(len(vocab))]
        print 'done data'
    
        return articles, date, vocab 
    
    
    def initParam(self, articles, date, vocab):  
        
        
        self.param['max_iter'] = 200
        self.param['T'] = 16
        self.param['D'] = len(articles)
        self.param['V'] = len(vocab)
        self.param['N'] = [len(article) for article in articles]
        self.param['alpha'] = [50.0/self.param['T'] for _ in range(self.param['T'])]
        self.param['beta'] = [0.1 for _ in range(self.param['V'])]
        self.param['beta_sum'] = sum(self.param['beta'])
        self.param['psi'] = [[1 for _ in range(2)] for _ in range(self.param['T'])]
        self.param['betafunc_psi'] = [scipy.special.beta(self.param['psi'][t][0], self.param['psi'][t][1] ) for t in range(self.param['T'])]
        self.param['word_id'] = {vocab[i]: i for i in range(len(vocab))}
        self.param['word_token'] = vocab
        self.param['t'] = [[date[d] for _ in range(self.param['N'][d])] for d in range(self.param['D'])]
        self.param['w'] = [[self.param['word_id'][articles[d][i]] for i in range(self.param['N'][d])] for d in range(self.param['D'])]
        self.param['m'] = [[0 for t in range(self.param['T'])] for d in range(self.param['D'])]
        self.param['n'] = [[0 for _ in range(self.param['V'])] for t in range(self.param['T'])]
        self.param['z'] = [[] for _ in range(self.param['D'])]
        
        [[self.param['z'][d].append(random.choice(self.param['topicseed'][word_id])) if self.param['topicseed'][word_id] \
          else self.param['z'][d].append(random.choice(self.param['topiclabel'][d])) if self.param['topiclabel'][d] \
          else self.param['z'][d].append(random.randrange(0, self.param['T'])) for word_id in self.param['w'][d]] for d in range(self.param['D'])]                
        
        self.param['n_sum'] = [0 for t in range(self.param['T'])]
        np.set_printoptions(threshold=np.inf)
        np.seterr(divide='ignore', invalid='ignore')
        self.updateParam(self.param)
        print 'done param'
        return self.param
    
    #method to initialize seed words for respective topics.     
    def init_seedwords(self, seed_file, vocab):
        temp_param = {}
        
        temp_param['word_id'] = {vocab[i]: i for i in range(len(vocab))}
        #self.param['topicseed'] = [[] for _ in range(len(vocab))]
        with open(seed_file) as seedfile:
            for lines in seedfile:
                line = lines.rstrip().split(' ')
                
                topic = line[0]
                print topic
                topic_id = self.addtopic(topic)
                print topic_id
                
                seedwords = [sw.lower() for sw in line[1].split(',')]
                stemmed_seed = self.stemArticle(seedwords)
                print stemmed_seed
                
                for seedword in stemmed_seed:
                    seedword_id = temp_param['word_id'].get(seedword,-1)
                    if seedword_id == -1:
                        print 'Word is not recognised'
                        continue
                    if topic_id not in self.param['topicseed'][seedword_id]:
                        self.param['topicseed'][seedword_id].append(topic_id)
                        print seedword
                        print self.param['topicseed'][seedword_id]
                    
    def updateParam(self, param):
        
        for d in range(param['D']):
            for i in range(param['N'][d]):
                topic_di = param['z'][d][i]        #topic in doc d at position i
                word_di = param['w'][d][i]        #word ID in doc d at position i
                param['m'][d][topic_di] += 1
                param['n'][topic_di][word_di] += 1
                param['n_sum'][topic_di] += 1
        
    def getTopicDate(self, param):
        
        self.param['topic_date'] = []
        for topic in range(param['T']):
            current_topic_date = []
            #assign the same date for all words of a topic in the same document
            current_topic_doc_date = [[ (param['z'][d][i]==topic)*param['t'][d][i] for i in range(param['N'][d])] for d in range(param['D'])]
            for d in range(param['D']):
                current_topic_doc_date[d] = filter(lambda x: x!=0, current_topic_doc_date[d])
            for date in current_topic_doc_date:
                current_topic_date.extend(date)
            assert current_topic_date != []
            self.param['topic_date'].append(current_topic_date)
        return self.param['topic_date']
    
    def GetMethodOfMomentsEstimatesForPsi(self, param):
        topic_date = self.getTopicDate(param)
        psi = [[1 for _ in range(2)] for _ in range(len(topic_date))]
        for i in range(len(topic_date)):
            current_topic_date= topic_date[i]
            date_mean = np.mean(current_topic_date)
            date_var = np.var(current_topic_date)
            if date_var == 0:
                date_var = 1e-6
            common_factor = date_mean*(1-date_mean)/date_var - 1
            psi[i][0] = 1 + date_mean*common_factor
            psi[i][1] = 1 + (1-date_mean)*common_factor
        return psi
    
    def ComputePosteriorEstimatesOfThetaAndPhi(self, param):
        theta = copy.deepcopy(param['m'])
        phi = copy.deepcopy(param['n'])

        for d in range(param['D']):
            if sum(theta[d]) == 0:
                theta[d] = np.asarray([1.0/len(theta[d]) for _ in range(len(theta[d]))])
            else:
                theta[d] = np.asarray(theta[d])
                theta[d] = 1.0*theta[d]/sum(theta[d])
        theta = np.asarray(theta)

        for t in range(param['T']):
            if sum(phi[t]) == 0:
                phi[t] = np.asarray([1.0/len(phi[t]) for _ in range(len(phi[t]))])
            else:
                phi[t] = np.asarray(phi[t])
                phi[t] = 1.0*phi[t]/sum(phi[t])
        phi = np.asarray(phi)

        return theta, phi
    
    
    def TopicsOverTimeGibbsSampling(self, param):
        
        for iteration in range(param['max_iter']):
            for d in range(param['D']):
                for i in range(param['N'][d]):
                    word_di = param['w'][d][i]
                    t_di = param['t'][d][i]

                    old_topic = param['z'][d][i]
                    param['m'][d][old_topic] -= 1
                    param['n'][old_topic][word_di] -= 1
                    param['n_sum'][old_topic] -= 1

                    topic_probabilities = []
                    for topic_di in range(param['T']):
                        psi_di = param['psi'][topic_di]
                        topic_probability = 1.0 * (param['m'][d][topic_di] + param['alpha'][topic_di])
#                         print "topic_prob 1 = %f" % (topic_probability)
                        a = ((1-t_di)**(psi_di[0]-1)) * ((t_di)**(psi_di[1]-1))
                        topic_probability *= a
#                         print "a = %f" % (a)
#                         print "topic_prob 2 = %f" % (topic_probability)
                        topic_probability /= param['betafunc_psi'][topic_di]
                        topic_probability *= (param['n'][topic_di][word_di] + param['beta'][word_di])
                        topic_probability /= (param['n_sum'][topic_di] + param['beta_sum'])
                        topic_probabilities.append(topic_probability)
                    sum_topic_probabilities = sum(topic_probabilities)
                    if sum_topic_probabilities == 0:
                        topic_probabilities = [1.0/param['T'] for _ in range(param['T'])]
                    else:
                        topic_probabilities = [p/sum_topic_probabilities for p in topic_probabilities]
                    new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1) #sample topic from multinomial distribution. return the index of sampled topic
                    #print new_topic
                    
                    if param['topicseed'][word_di]:
                        new_topic = np.random.choice(param['topicseed'][word_di])
                        param['z'][d][i] = new_topic
                        param['m'][d][new_topic] += 1
                        param['n'][new_topic][word_di] += 1
                        param['n_sum'][new_topic] += 1
                    else:
                        param['z'][d][i] = new_topic
                        param['m'][d][new_topic] += 1
                        param['n'][new_topic][word_di] += 1
                        param['n_sum'][new_topic] += 1

                if d%1000 == 0:
                    print('Done with iteration {iteration} and document {document}'.format(iteration=iteration, document=d))
            param['psi'] = self.GetMethodOfMomentsEstimatesForPsi(param)
            param['betafunc_psi'] = [scipy.special.beta( param['psi'][t][0], param['psi'][t][1] ) for t in range(param['T'])]
        param['m'], param['n'] = self.ComputePosteriorEstimatesOfThetaAndPhi(param)
        return param['m'], param['n'], param['psi']


        
            