#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:36:59 2024

@author: rdx
"""

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    # Create a 2D array to store lengths of longest common subsequence
    dp = [[0] * (n + 1) for i in range(m + 1)]

    # Build the dp array in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Following code is used to print LCS
    index = dp[m][n]

    # Create a character array to store the lcs string
    lcs_str = [""] * (index+1)
    lcs_str[index] = ""

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs_str
    i = m
    j = n
    while i > 0 and j > 0:
        # If current character in X[] and Y[] are same, then
        # current character is part of LCS
        if X[i-1] == Y[j-1]:
            lcs_str[index-1] = X[i-1]
            i -= 1
            j -= 1
            index -= 1
        # If not same, then find the larger of two and
        # go in the direction of the larger value
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    print("LCS of", X, "and", Y, "is", "".join(lcs_str))

# Example usage
X = "AGGTAB"
Y = "GXTXAYB"
lcs(X, Y)



# jaro-winkler
from fuzzywuzzy import fuzz
fuzz.ratio("kitten", "sitting")  # Output: Similarity percentage


    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Example usage
print(cosine_sim("This is a sample text.", "This text is a sample."))  # Output: Similarity score


def n_gram_similarity(s1, s2, n):
    def n_grams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    
    grams1 = n_grams(s1, n)
    grams2 = n_grams(s2, n)
    return len(grams1.intersection(grams2)) / len(grams1.union(grams2))

# Example usage
print(n_gram_similarity("kitten", "sitting", 2))  # Output: Similarity score
