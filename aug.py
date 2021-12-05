import os
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import copy
import itertools
from scipy import spatial


"""
PPDB Class
"""
PPDB_MODEL = {}

class Ppdb():
    # http://paraphrase.org/#/download
    def __init__(self, dict_path):
        super(Ppdb, self).__init__()

        self.cache = True
        self.dict_path = dict_path
        self.lang = 'eng'  # TODO: support other languages

        self.score_threshold = self.get_default_score_thresholds() # TODO: support other filtering
        self.is_synonym = True  # TODO: antonyms

        self._init()

    def _init(self):
        self.dict = {}
        self.read(self.dict_path)

    @classmethod
    def get_default_score_thresholds(cls):
        return {
            'AGigaSim': 0.6
        }

    def read(self, model_path):
        with open(model_path, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')

                if '\\ x' in line or 'xc3' in line:
                    continue

                fields = line.split('|||')
                constituents = fields[0].strip()[1:-1].split('/')
                phrase = fields[1].strip()
                paraphrase = fields[2].strip()

                # filter multiple words
                if len(phrase.split()) != len(paraphrase.split()):
                    continue

                scores = []

                if len(fields) == 6:
                    # filter equivalence word ( for PPDB v2.0 only.)
                    # entailment = fields[5].strip()
                    # if entailment == 'Equivalence' and self.is_synonym:
                    #     continue

                    features = fields[3].strip().split()
                    features = [feature for feature in features for s in self.score_threshold if
                                s in feature]  # filter by scheme

                    for feature in features:
                        scheme, score = feature.split('=')
                        if scheme in self.score_threshold and float(score) > self.score_threshold[scheme]:
                            scores.append((scheme, score))

                    # # filter by feature/ score
                    # if len(scores) == 0:
                    #     continue

                if phrase not in self.dict:
                    self.dict[phrase] = {}

                part_of_speeches = [con for con in constituents ]

                for pos in part_of_speeches:
                    if pos not in self.dict[phrase]:
                        self.dict[phrase][pos[0:1]] = []

                    self.dict[phrase][pos[0:1]].append({
                        'pos':pos,
                        'phrase': phrase,
                        'synonym': paraphrase,
                        'scores': scores
                    })

    def predict(self, word, pos=None):
        if pos is None:
            candidates = []
            if word not in self.dict:
                return candidates

            for pos in self.dict[word]:
                for candidate in self.dict[word][pos]:
                    candidates.append(candidate['synonym'])

            return candidates

        if word in self.dict and pos in self.dict[word]:
            return [candidate['synonym'] for candidate in self.dict[word][pos]]

        return []

    def pos_tag(self, tokens):
        return nltk.pos_tag(tokens)


def main():
    
    """
    Load PPDB class
    """
    with open('dataset/ppdb_model.bin','rb') as f:
        ppdb_model = pickle.load(f)
        

    """
    Universal Sentence Encoder module load
    """

    enc_model_tfr = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
    # enc_model_tfr = hub.load('model/universal-sentence-encoder-large_5')


    """
    Cosine similarity function for vectors
    """
    def cosine_similarity(vector1, vector2):
        return 1 - spatial.distance.cosine(vector1, vector2)



    """
    Load input Dataset
    """

    indf = pd.read_csv('dataset/labeled_messages.csv', header=0)
    indf.drop('ignore', 1, inplace=True)
    indf.dropna(inplace=True)
    indf.replace(r"^ +| +$", r"", regex=True, inplace=True)

    indf_empathy_list = (indf.astype(str).assign(empathy=indf.empathy.str.split(','))
                            .explode('empathy')
                            .reset_index()).empathy.tolist()
    indf_empathy_list = [x for x in indf_empathy_list if str(x) != 'nan']
    indf_empathy_list = [x.lower().strip() for x in indf_empathy_list]
    indf_empathy_list = list(set(indf_empathy_list))
    indf_empathy_list.sort()




    """
    Stop words and ppdb_model words
    """

    stop = set(stopwords.words('english') + list(string.punctuation))
    ppdb_model_dict_keys = list(ppdb_model.dict.keys())

    
    """

    # 1. Find most similar word in the sentence w.r.t target
    # 2. Replace that word with the synonym
    # 3. Get the cosine similarity (using Universal Snetence Encoder embeddings) of the new and input sentence. 
    # If the similarity is GT 0.7 then accept else reject
    """
    odf_list = []
    for ind in indf.index:
        odf_list.append(ind)
        
        message = indf['message'][ind].lower()
        odf_list.append(message)
        empathy = indf['empathy'][ind]
        odf_list.append(empathy)

        message_token = [i for i in word_tokenize(message.lower()) if (i not in stop and i in ppdb_model_dict_keys)] 

        if len(message_token) > 0 :


            message_token_vector = enc_model_tfr(message_token)

            empathy_list = [x.strip() for x in str(empathy).lower().split(',') if str(x) != 'nan']

            sim_words_master_list = []

            for empathy_item in empathy_list:
                
                # 1. Find most similar word in the sentence w.r.t target
                
                empathy_item_vector = enc_model_tfr([empathy_item])

                empathyItem_message_vector_rel = np.array(empathy_item_vector).dot(np.array(message_token_vector).T)

                empathyItem_message_vector_highest_sim_idx = np.argmax(empathyItem_message_vector_rel)

                message_sim_word = message_token[empathyItem_message_vector_highest_sim_idx]

                try:

                    ppdb_dict_value = ppdb_model.predict( word= message_sim_word, pos=None )
                    ppdb_dict_value = list(set(ppdb_dict_value))
                except:
                    
                    ppdb_dict_value= []
                    
                    
                sim_words_list = []
        
                for ppdb_dict_value_item in ppdb_dict_value:

                    if ((ppdb_dict_value_item.strip()  == empathy_item) or \
                        (ppdb_dict_value_item.strip()  not in indf_empathy_list)) and \
                            (message_sim_word.strip() != ppdb_dict_value_item.strip()):
                        
                        sim_words_list.append(message_sim_word.strip() + '##'+ ppdb_dict_value_item.strip())
                        
                if len(sim_words_list) > 0:
                    sim_words_master_list.append(sim_words_list)


            #Generate combinations of similar words w.r.t. input words
            sim_words_combinations = list(itertools.product(*sim_words_master_list))

            for sim_words_combinations_item in sim_words_combinations:
                
                new_message = copy.deepcopy(message)
                
                # 2. Replace that word with the synonym
                for sim_words_combinations_item_word in sim_words_combinations_item:
                    
                    wordComb = sim_words_combinations_item_word.split('##')
                    new_message = new_message.replace(wordComb[0], wordComb[1])
                
                
                # 3. Get the cosine similarity of the new and input sentence. If the similarity is GT 0.7 then accept wlse reject
                message_vector = enc_model_tfr([message])
                new_message_vector = enc_model_tfr([new_message])
                
                oldmessage_newMessage_vector_rel = cosine_similarity(message_vector, new_message_vector) 
                
                if oldmessage_newMessage_vector_rel > 0.7:
                    odf_list.append(ind)
                    
                    odf_list.append(new_message)
                    # print(new_message)

                    odf_list.append(empathy)                   
                    
            print(ind , ' message : ', message, '. | new messages count', int((len(odf_list)/3)-int(ind)))

            # new_df = pd.DataFrame(np.array(odf_list).reshape(-1,3), columns = ['old_id','message','empathy'])
            
            # new_df.to_csv('dataset/new_labeled_messages.csv',index_label='index')
            
    new_df = pd.DataFrame(np.array(odf_list).reshape(-1,3), columns = ['old_id','message','empathy'])

    new_df.to_csv('dataset/new_labeled_messages.csv',index_label='index')


if __name__ == "__main__":
    main()


