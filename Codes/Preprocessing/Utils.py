import numpy as np
import pandas as pd
import re



def clean_reviews(review):
    """
    This function will remove the unncessary 
    symbols from a review such as punctuation mark, numbers ,emoji.
    """

    review = review.replace('\n','') #removing new line 
    review = re.sub('[^\u0980-\u09FF]',' ',str(review)) #removing unnecessary punctuation
    return review

def get_stopwords(filename):
    """
    This function will create a stopwords list from the (.txt) file. 
    """
    stp_words = open(filename,'r',encoding='utf-8').read().split()
    num_of_stopwords = len(stp_words)
    return stp_words,num_of_stopwords

def remove_stopwords(review,filename):
    """
    This function will remove the stopwords from a review. 
    """
    stp_words,num_of_stopwords =get_stopwords(filename)
    result = review.split()
    reviews = [word.strip() for word in result if word not in stp_words ]
    reviews =" ".join(reviews)
    return reviews    
