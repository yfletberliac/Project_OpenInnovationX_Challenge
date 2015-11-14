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
from textstat.textstat import textstat
import matplotlib.patches as mpatches

colours = []
colours.append( (26, 188, 156))
colours.append( (52, 152, 219))
colours.append( (155, 89, 182))
colours.append( (241, 196, 15))
colours.append( (231, 76, 60))
colours.append( (46, 204, 113))
colours.append( (230, 126, 34))
colours.append( (149, 165, 166))
colours.append( (52, 73, 94))

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


def calculateSentiment( tokens ):
	file = open('Data_Set_S1.txt', 'r')
	sentiments_dataset = file.readlines()
	word_sentiment = dict()
	for l in sentiments_dataset:
		if len(l) and l[0] != '!':
			ll = l.split('\t')
			word_sentiment[ll[0]] = float(ll[2])
	s = 0.0
	keys = set(word_sentiment.keys())
	total = 0
	for t in tokens:
	    if t in keys:
	        s += word_sentiment[t]
	        total +=1
	if s:
	    return float(s)/float(total)
	else:
	    return 0.0

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

def complexityAlongtheText( f, chunk_length = 100 ):
	text = loadText(f)
	words = text.split()
	x = []
	y = []
	cur = 0
	average = textstat.flesch_reading_ease(text)

	while ( cur < len(words) ):
	    sub = words[cur:cur+chunk_length]
	    sub.append('.')
	    sub_text = ' '.join(sub)
	    y.append( 100 - textstat.flesch_reading_ease(sub_text)  )
	    x.append( cur)
	    cur += chunk_length
	    
	if average < 20:
	    col = colours[4]
	elif average < 40:
	    col = colours[6]
	elif average < 60:
	    col = colours[3]
	elif average < 80:
	    col = colours[1]
	else:
	    col = colours[0]
	plt.plot(x,y, color = [ 1.0 / 255.0 * c for c in col], alpha = 0.6, linewidth = 5)    
	plt.fill_between(x,y, color = [ 1.0 / 255.0 * c for c in col], alpha = 0.3)
	#     plt.plot( [0,max(x)], [average,average], color = 'gray')
	plt.ylim([0,100])
	plt.xlabel("number of words")
	plt.ylabel("difficulty")
	plt.show()


def getTopicsDistributionWithinTheText(path, chunk_length = 300 ):
    
	global_scores = topicsFromTokens(tokenize(loadText(path)))
	text = loadText(path)
	words = text.split()
	average = textstat.flesch_reading_ease(text)
	scores = dict()
	for i in sorted(global_scores, key=lambda tup: tup[1], reverse = True):
	    if i[1] > min_score:
	        scores[i[0]] = []
	x = []
	y = []
	cur = 0
	i = 1
	while ( cur < len(words) ):
	    sub = words[cur:cur+chunk_length]
	    sub.append('.')
	    sub_text = ' '.join(sub)
	    cur += chunk_length
	    
	    bow = lda.id2word.doc2bow(raw_tokenize(sub_text))
	    score = lda.get_document_topics(bow)
	    for s in score:
	        if s[0] in scores.keys():
	            scores[s[0]].append(s[1])
	            
	    for s in scores:
	        if len(scores[s]) < i:
	            scores[s].append(0)
	    i += 1
	    
	    
	return scores, global_scores



def displayTopicsDistributionWithinTheText(f, chunk_length = 300, dispLabels = True,pie = False):
	topic_names = pickle.load( open( "topics_names.p", "rb" ) )
	distribAlongText, global_scores = getTopicsDistributionWithinTheText(f, chunk_length)

	global_scores =  sorted(global_scores, key=lambda tup: tup[1], reverse = True)
	scores = []
	for g in global_scores:
	    if g[0] in distribAlongText.keys():
	        scores.append( distribAlongText[g[0]])

	values = []
	labels = []
	somme= 0.
	for s in global_scores:
	    if s[1] > min_score:
	        values.append(s[1])
	        somme+= s[1]**2
	        labels.append(topic_names[s[0]])

	values = [ float(v)/float(somme) for v in values]

	if pie:
	    # draw pie chart
	    matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)  
	    plt.pie(values,  labels=labels, colors = [ [1.0 / 255.0 * c for c in cc] for cc in colours[1:]])
	    plt.show()


	matplotlib.rcParams['figure.figsize'] = (15.0, 8.0)
	patches = []    
	x = range( 0, len(scores[0]))
	x = [ chunk_length * i for i in x]
	i = 0
	k = 1
	for s in scores:
		c = [ 1.0 / 255.0 * c for c in colours[k] ]
		plt.plot(x, s,linewidth = 5, color = c, alpha = 0.6)
		plt.fill_between(x, s,linewidth = 5, color = c, alpha = 0.3)
		patches.append(mpatches.Patch(color=c, label=labels[i]))
		k += 1
		i +=1
		if k >= len(colours):
		    k = 0
	        
	if dispLabels:
	    plt.legend(handles=patches)
	    plt.ylabel('proportion')
	    plt.xlabel('number of words')
	else:
	    plt.axis('off')

	plt.show()






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


def LouvainModularity( G, colours, node_size = 80):

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

	    nx.draw_networkx_nodes(G, pos, list_nodes, linewidths = 0,  node_size = node_size,
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


def displayResults( path ):
	print "stats"
	text = loadText(path)
	raw_tokens = raw_tokenize(text)
	print "number of words %s" %count_words(text)
	print "number of sentences %s" %textstat.sentence_count(text)
	print "uniques words: %s" %len(set(raw_tokenize(text)))
	print "Difficulty %s / 100 " %(100 - textstat.flesch_reading_ease(text))
	print "Average sentiment %s (negative: 0, neutral: 5, positive: 10)"%calculateSentiment(raw_tokens)
	print
	print "topic distribution"
	displayTopicsDistributionWithinTheText(path, 300, pie = False)
	print "difficulty over the text "
	complexityAlongtheText( path, 300)