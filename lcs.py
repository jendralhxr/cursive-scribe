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


def lcs_multiple(*strings):
    from functools import reduce
    from itertools import product

    # Determine the lengths of all input strings
    lengths = [len(s) for s in strings]
    num_strings = len(strings)

    # Create a (num_strings + 1)-dimensional array to store lengths of LCS
    dp = reduce(lambda x, y: x + [0], [[0] * (length + 1) for length in lengths], [])

    # Build the dp array in a generalized fashion
    for indices in product(*[range(length + 1) for length in lengths]):
        if any(index == 0 for index in indices):
            continue
        prev_indices = tuple(index - 1 for index in indices)
        if all(strings[i][indices[i] - 1] == strings[0][indices[0] - 1] for i in range(num_strings)):
            dp[indices] = dp[prev_indices] + 1
        else:
            dp[indices] = max(dp[indices[:i] + (indices[i] - 1,) + indices[i + 1:]] for i in range(num_strings))

    # Reconstruct the LCS from the dp array
    indices = tuple(lengths)
    lcs_str = []

    while all(index > 0 for index in indices):
        prev_indices = tuple(index - 1 for index in indices)
        if all(strings[i][indices[i] - 1] == strings[0][indices[0] - 1] for i in range(num_strings)):
            lcs_str.append(strings[0][indices[0] - 1])
            indices = prev_indices
        else:
            max_index = max((i for i in range(num_strings)), key=lambda i: dp[indices[:i] + (indices[i] - 1,) + indices[i + 1:]])
            indices = indices[:max_index] + (indices[max_index] - 1,) + indices[max_index + 1:]

    print("LCS of", strings, "is", "".join(reversed(lcs_str)))

# Example usage
strings = ["AGGTAB", "GXTXAYB", "AGXGTAYB"]
lcs_multiple(*strings)






# jaro-winkler
from fuzzywuzzy import fuzz
fuzz.ratio("kitten", "sitting")  # Output: Similarity percentage

def jaro_distance(s1, s2):
    if s1 == s2:
        return 1.0

    len_s1 = len(s1)
    len_s2 = len(s2)

    max_dist = int(max(len_s1, len_s2) / 2) - 1

    match = 0

    hash_s1 = [0] * len_s1
    hash_s2 = [0] * len_s2

    for i in range(len_s1):
        for j in range(max(0, i - max_dist), min(len_s2, i + max_dist + 1)):
            if s1[i] == s2[j] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    if match == 0:
        return 0.0

    t = 0
    point = 0

    for i in range(len_s1):
        if hash_s1[i]:
            while hash_s2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1

    t /= 2

    return (match / len_s1 + match / len_s2 + (match - t) / match) / 3.0


def jaro_winkler_distance(s1, s2, p=0.1):
    jaro_dist = jaro_distance(s1, s2)

    prefix = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    prefix = min(4, prefix)

    return jaro_dist + (prefix * p * (1 - jaro_dist))


# Example usage
s1 = "CRATE"
s2 = "TRACE"
print(f"Jaro-Winkler distance between '{s1}' and '{s2}' is {jaro_winkler_distance(s1, s2)}")


#----
# one with keyerror
from itertools import product

def lcs_multiple(*strings):
    if not strings:
        return []

    num_strings = len(strings)
    lengths = [len(s) for s in strings]

    # Create a multi-dimensional array to store lengths of LCS
    dp = {}
    for indices in product(*(range(length + 1) for length in lengths)):
        dp[indices] = (0, set())

    # Build the dp array in bottom-up fashion
    for indices in product(*(range(length + 1) for length in lengths)):
        if all(index == 0 for index in indices):
            continue

        max_length = 0
        max_subseqs = set()
        for i in range(num_strings):
            if indices[i] > 0:
                prev_indices = indices[:i] + (indices[i] - 1,) + indices[i + 1:]
                if dp[prev_indices][0] > max_length:
                    max_length = dp[prev_indices][0]
                    max_subseqs = dp[prev_indices][1]

                elif dp[prev_indices][0] == max_length:
                    max_subseqs |= dp[prev_indices][1]

        current_chars = [strings[i][indices[i] - 1] for i in range(num_strings)]
        if all(char == current_chars[0] for char in current_chars):
            prev_indices = tuple(index - 1 for index in indices)
            if dp[prev_indices][0] + 1 > max_length:
                max_length = dp[prev_indices][0] + 1
                max_subseqs = {subseq + current_chars[0] for subseq in dp[prev_indices][1] or {""}}

            elif dp[prev_indices][0] + 1 == max_length:
                max_subseqs |= {subseq + current_chars[0] for subseq in dp[prev_indices][1] or {""}}

        dp[indices] = (max_length, max_subseqs)

    # The LCS is in dp[lengths]
    _, lcs_set = dp[tuple(lengths)]
    return list(lcs_set)

# Example usage
strings = ["AGGTAB", "GXTXAYB", "AGXGTAYB"]
lcs_results = lcs_multiple(*strings)
print("LCS results:", lcs_results)

#------
# one which works
from itertools import product

def lcs_multiple(*strings):
    if not strings:
        return []

    num_strings = len(strings)
    lengths = [len(s) for s in strings]

    # Create a multi-dimensional array to store lengths of LCS and sets of LCS substrings
    dp = {}
    for indices in product(*(range(length + 1) for length in lengths)):
        dp[indices] = (0, {""})

    # Build the dp array in bottom-up fashion
    for indices in product(*(range(length + 1) for length in lengths)):
        if all(index == 0 for index in indices):
            continue

        max_length = 0
        max_subseqs = set()
        for i in range(num_strings):
            if indices[i] > 0:
                prev_indices = indices[:i] + (indices[i] - 1,) + indices[i + 1:]
                if dp[prev_indices][0] > max_length:
                    max_length = dp[prev_indices][0]
                    max_subseqs = dp[prev_indices][1]

                elif dp[prev_indices][0] == max_length:
                    max_subseqs |= dp[prev_indices][1]

        current_chars = [strings[i][indices[i] - 1] for i in range(num_strings) if indices[i] > 0]
        if len(current_chars) == num_strings and all(char == current_chars[0] for char in current_chars):
            prev_indices = tuple(index - 1 for index in indices)
            if dp[prev_indices][0] + 1 > max_length:
                max_length = dp[prev_indices][0] + 1
                max_subseqs = {subseq + current_chars[0] for subseq in dp[prev_indices][1] or {""}}

            elif dp[prev_indices][0] + 1 == max_length:
                max_subseqs |= {subseq + current_chars[0] for subseq in dp[prev_indices][1] or {""}}

        dp[indices] = (max_length, max_subseqs)

    # The LCS is in dp[lengths]
    _, lcs_set = dp[tuple(lengths)]
    return list(lcs_set)

# Example usage
strings = ["AGGTAB", "GXTXAYB", "AGXGTAYB"]
lcs_results = lcs_multiple(*strings)
print("LCS results:", lcs_results)
