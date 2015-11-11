# imports
from __future__ import division
import logging, gensim, bz2
from gensim import corpora, models, similarities
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from bs4 import BeautifulSoup
import nltk
import re
from nltk.corpus import stopwords
import string
import itertools
import random
exclude = set(string.punctuation)
import matplotlib
matplotlib.rcParams['figure.figsize'] = (18.0, 18.0)
import numpy as np
from lightning import Lightning
from numpy import random, asarray
import networkx as nx
import math
import random
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
import os
import sys
import PIL
from PIL import Image
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.sparse import *
from scipy import *
import community

# constants
min_score = 0.05

# load lda
lda = gensim.models.ldamodel.LdaModel.load(u'lda/wikipedia_lda', mmap='r')

# load text xliff or txt format
def loadText(path):   
    if path[-4:] == 'liff':
        soup = BeautifulSoup( open(path), 'lxml')
        s = ' '
        for string in soup.find_all("source"):
            s += ' ' + string.string
        return s
    else:
        return open(path, 'r').read()



# raw tokenize
def raw_tokenize(text):
    text = text.lower()
    # tokenize + punctuation
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
    return tokenizer.tokenize(text)


# tokenize
def tokenize(text):
    text = raw_tokenize(text)
    # remove stopwords
    from nltk.corpus import stopwords
    stops = stopwords.words('english')
    text = [ w for w in text if w.lower() not in stops]
    # Exclude numbers
    text = [s for s in text if not re.search(r'\d',s)]
    #remove word with less than 3letters
    text = [s for s in text if len(s) > 2]
    # stemmer
    lmtzr = WordNetLemmatizer()
    text =  [(lmtzr.lemmatize(t)) for t in text] 
    return text

# count words
def count_words(text):
    text = text.lower()
    # tokenize + punctuation
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
    text = tokenizer.tokenize(text)
    
    return len(text)

# get filelist
def getFileList(root):
	filelist = []
	for filename in os.listdir(root):
		filename = root + filename
		filelist.append(filename.encode(sys.getfilesystemencoding()))

	return filelist


# get topics froms tokens
def topicsFromTokens( tokens ):
    bow = lda.id2word.doc2bow( tokens )
    return lda.get_document_topics(bow)


# print word cloud
def WordCloudTopic( items , imagePath = None):
    # Generate a word cloud image
    
    if imagePath:
    	alice_coloring = np.array(Image.open(imagePath))

    	wc = WordCloud(background_color="white", max_words=200, mask=alice_coloring,
                   stopwords=STOPWORDS.add("said"),
                   max_font_size=300)
    	# generate word cloud
    	wc.generate_from_frequencies(items)
    	image_colors = ImageColorGenerator(alice_coloring)
    	plt.imshow(wc.recolor(color_func=image_colors))
    else:
    	wc = WordCloud(background_color="white", max_words=300,
        max_font_size=40, random_state=42)
    	wordcloud = wc.generate_from_frequencies(items)    
    	plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

## similarity
# get topic from a single word
def getTopicFromWord( unique_word ):
    bow = lda.id2word.doc2bow( [unique_word] )
    topic = lda.get_document_topics(bow)
    if len(topic):
        return topic[0][0]
    else:
        return None

def norm(v):
    s = 0
    for d in v.itervalues():
        s+= d**2
    return sqrt(s)

def similarity( a, b):
    # cosine similarity
    p = (a.dot(b.transpose()) / (norm(a) * norm(b))).data
    if len(p):
        return p[0]


def closestFile( path ,semantic_vectors, printit = False ):
    vec = topicsFromTokens( tokenize(loadText(path)))
    u = dok_matrix((1,100), dtype=float32)
    for t in vec:
        u[0,t[0]] = t[1]
    similarities = dict()
    for s in semantic_vectors:
        vec = semantic_vectors[s]
        v = dok_matrix((1,100), dtype=float32)
    	for t in vec:
        	v[0,t[0]] = t[1]
        similarities[s] = similarity(v,u)  
        
    if printit:
        k = 1
        for i in sorted(similarities.items(), key=lambda x: x[1])[::-1][:15]:
            print str(k) + "  |  "  + str(i)
            k += 1
        
    return similarities


class idGenerator:
    def __init__(self):
        self.id = 0
    def get(self):
        self.id += 1
        return self.id - 1


# convert raw semantic vector to sprse DOK vector
def semantic_vec_to_dok( semantic_vec , n_topics = 100):
    u = dok_matrix((1,n_topics), dtype=float32)
    for t in semantic_vec:
        u[0,t[0]] = t[1]
    return u

# draw graph 
def getGraph(semantics_vectors, color, similarity_cutoff = 0.85):
	filelist = semantics_vectors.keys()
	id_gen  = idGenerator()
	graph = nx.Graph()
	filename2id = dict()
	base_weight = 5.0

    # construct nodes
	for f in filelist:
	    i = id_gen.get()
	    filename2id[f] = i
	    filename2id[i] = f
	    graph.add_node( i , weight = base_weight, label = f.decode('unicode_escape').encode('ascii','ignore'), color = color)

	# construct edges
	for f in filelist:  
	    i = filename2id[f]
	    u = semantic_vec_to_dok( semantics_vectors[f])
	    for ff in filelist:
	        if ff != f:
	            ii = filename2id[ff]
	            v = semantic_vec_to_dok(semantics_vectors[ff])
	            sim = similarity(u,v)
	            if sim > similarity_cutoff:
	                distance = 1 - sim
	                if distance < 0:
	                	distance = 0
	                graph.add_edge(i, ii, weight = distance)
	                
	return graph, filename2id



def plotGraphNX( G ):
	matplotlib.rcParams['figure.figsize'] = (30.0, 22.0)
	groups = []
	i = 0
	tresh = 0.1
	while i <1:
	    groups.append((i , [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <i and d['weight'] >= i - tresh ]))
	    i += tresh
	pos=nx.spring_layout(G) # positions for all nodes
	# nodes
	nx.draw_networkx_nodes(G,pos,node_size=400, node_color = 'orange', alpha = 0.6)
	# edges
	for g in groups:
	    nx.draw_networkx_edges(G,pos,edgelist=g[1],
	                    width=1, alpha = g[0])
	# nx.draw_networkx_edges(G,pos,edgelist=esmall,width=6,alpha=0.5,edge_color='b',style='dashed')
	# labels
	# nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
	plt.axis('off')
	plt.show() 
	return pos

def computeModularity ( graph_undirect, sub_graphs):
	M = 0
	L = len( graph_undirect.edges() )
	for c in sub_graphs:
	    kc = 0
	    for n in c:
	        kc+= c.degree(n)
	    Lc = len( c.edges() )
	    M += ( float( Lc / float(L) - pow( float(kc) / ( 2.0 * float(L) ) ,2) ))
	return M


def LouvainModularity( G, colours):
	matplotlib.rcParams['figure.figsize'] = (30.0, 20.0)

	graphs = list(nx.connected_component_subgraphs(G))
	graphs.sort(key=lambda x: len( x.nodes() ), reverse = True)
	G = graphs[0]

	#first compute the best partition
	partition = community.best_partition(G)

	#drawing
	size = float(len(set(partition.values())))
	pos = nx.spring_layout(G)
	count = 0.
	nx.draw_networkx_edges(G,pos,  edge_color = "#aaaaaa", alpha = 0.8)
	sub_graphs_louvain = []

	k = 0
	for com in set(partition.values()) :
	    count = count + 1.
	    list_nodes = [nodes for nodes in partition.keys()
	                                if partition[nodes] == com]

	    nx.draw_networkx_nodes(G, pos, list_nodes, linewidths = 0,  node_size = 80,
	                                node_color = colours[k])
	    #place partition in subgraphs
	    sub_graphs_louvain.append( G.subgraph(list_nodes ) )
	    k += 1
	    if k >= len(colours):
	        k = 0


	plt.axis("off")
	plt.savefig('grap_communties_louvain.png')
	plt.show()

	print "modularity: %s" %computeModularity ( G, sub_graphs_louvain)
	print "%s communities" %len(sub_graphs_louvain)
	return sub_graphs_louvain