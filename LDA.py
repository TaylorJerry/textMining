# for this text mining exercise this tutorial has been used: 
# https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

# Importing modules
from numpy import append
import pandas as pd
#import os.chdir('..')

# Read data into papers old solution
#papers = pd.read_csv('cleaned.csv')# Print head
#papers.head()

# Read data into papers
import textract
import io
import os
data_all = os.listdir('./Papers')
papers = pd.DataFrame()

for i in data_all:
    path = f'./Papers/{i}'
    byte_text = textract.process(path)
    raw_text = str(byte_text,'utf-8')
    raw_text = raw_text.replace(',', '')
    raw_text = raw_text.replace('"', '')
    raw_text = io.StringIO(raw_text)
    df=pd.read_csv(raw_text, header=None)
    papers = papers.append(df)
    print(i)

papers['paper_text'] = papers[0]

# Load the regular expression library
import re

# Remove punctuation
papers['paper_text_processed'] = \
papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', str(x)))

# Convert the titles to lowercase
papers['paper_text_processed'] = \
papers['paper_text_processed'].map(lambda x: x.lower())

# Print out the first rows of papers
papers['paper_text_processed'].head()

# Import the wordcloud library
from wordcloud import WordCloud

# Join the different processed titles together.
long_string = ','.join(list(papers['paper_text_processed'].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()

wordcloud.to_file("wordcloud.png")

import gensim
from gensim.utils import simple_preprocess
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'et', 'al', 'ii', 'pp', 'et al', 'n'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
        if word not in stop_words] for doc in texts]
        
data = papers.paper_text_processed.values.tolist()

data_words = list(sent_to_words(data))

# remove stop words
data_words = remove_stopwords(data_words)

print(data_words[:1][0][:30])


import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])

from pprint import pprint

# number of topics
num_topics = 10

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
                                       
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
import os

# Visualize the topics
#pyLDAvis.enable_notebook()
#pyLDAvis.write_html("result.html")

LDAvis_data_filepath = os.path.join('results/ldavis_prepared2_'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself

if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared2_'+ str(num_topics) +'.html')
    
#LDAvis_prepared
