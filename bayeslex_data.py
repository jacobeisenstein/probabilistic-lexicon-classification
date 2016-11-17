import codecs
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np

'''
Utilities for loading data
'''

getTokenCounts = lambda line : [token.split(':') for token in line.split() if token.split(':')[0].isalpha()]

open = lambda filename, mode : codecs.open(filename,mode=mode,encoding='utf-8')

def loadData(prefix,max_vocab):
    """ load bag-of-words and key files, up to max_vocab
    
    Parameters
    ----------
    prefix: str
    max_vocab: int
    
    Returns
    -------
    labels: np.array
    x: csr_matrix
    vocab: doct
    """
    
    vocab = {}

    docs = []
    words = []
    counts = []
    labels = []
    t = 0

    vocab_counter = Counter()
    
    with open(prefix+'.bow','r') as fin_bow:
        for line in fin_bow:
            for word,count in getTokenCounts(line):
                try:
                    vocab_counter[word] += int(count)
                except:
                    pass
    vocab = {j[0]:i for i,j in enumerate(vocab_counter.most_common(min(max_vocab,len(vocab_counter.keys()))))}
    
    with open(prefix+".key",'r') as fin_key:
        with open(prefix+".bow",'r') as fin_bow:
            for key_line in fin_key:
                bow_line = fin_bow.readline()
                label_str = key_line.rstrip()[-3:]
                label = None
                
                if label_str == 'POS':
                    label = 1.
                elif label_str == 'NEG':
                    label = -1.
                else:
                    raise ValueError("%s is not a valid label"%(label))
                if label is not None:
                    labels.append(label)
                    for word,count in getTokenCounts(bow_line):
                    #for word,count in [token.split(':') for token in bow_line.split()]:
                        try:
                            if word in vocab:
                                docs.append(t)
                                words.append(vocab[word])
                                counts.append(int(count))
                        except:
                            pass
                    t+=1

    x = csr_matrix((counts,(docs,words)),shape=(docs[-1]+1,len(vocab))).astype('float')
    return np.array(labels), x, vocab

def loadExtraData(filename,vocab):
    """ load extra bag-of-words, given existing vocabulary
    
    Parameters
    ----------
    filename: str
    vocab: dict
    
    Returns
    -------
    x: csr_matrix
    """
    
    counts = []
    docs = []
    words = []
    with open(filename,'r') as fin:
        for i,line in enumerate(fin):
            for word,count in getTokenCounts(line):
                if word in vocab:
                    docs.append(i)
                    words.append(vocab[word])
                    counts.append(int(count))
    return csr_matrix((counts,(docs,words)),shape=(docs[-1]+1,len(vocab))).astype('float')
                    
def getLex(lexfile,vocab):
    """ get a lexicon, given a file and a vocabulary dict
    
    Parameters
    ----------
    lexfile: name of a file containing a lexicon
    vocab: a dict of words to indices
    
    Returns
    -------
    x: sorted list of word indices
    """

    '''
    Inputs:
    
    - lexfile: name of a file containing a lexicon
    - a dict of words to indices
    
    Outputs:
    - a sorted list of word indices from the lexicon
    '''
    with open(lexfile,'r') as fin:
        words = fin.readlines()
        return sorted([vocab[word.rstrip()] for word in words if word.rstrip() in vocab])

