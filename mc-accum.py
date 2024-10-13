#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:22:03 2024
@author: jendralhxr
"""

def lcs(str1, str2, m, n):
    if m==0 or n==0:
        return 0 
    elif str1[m-1] == str2[n-1]: 
        return 1+lcs(str1, str2, m-1, n-1) 
    else: 
        return max(lcs(str1, str2, m-1, n),lcs(str1, str2, m,n-1))


##### MC

import random
import difflib


def string_similarity(str1, str2):
    """
    This function computes a similarity score between two strings.
    A value between 0 and 1, where 1 means exact match.
    """
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def monte_carlo_cumulative_match(target_string, groups, trials=1000):
    """
    Monte Carlo search to accumulate similarity scores for each group.
    
    Args:
    - target_string: The string to be matched.
    - groups: A dictionary where each key is a group name and each value is a list of substrings.
    - trials: Number of random trials to perform.
    
    Returns:
    - cumulative_scores: A dictionary with the cumulative similarity score for each group.
    """
    cumulative_scores = {group_name: 0 for group_name in groups.keys()}
    counts = {group_name: 0 for group_name in groups.keys()}  # To track how many times each group is sampled
    
    for _ in range(trials):
        # Randomly pick a group and a substring from that group
        group_name = random.choice(list(groups.keys()))
        substring = random.choice(groups[group_name])
        
        # Compute similarity between the target string and the chosen substring
        score = string_similarity(target_string, substring)
        
        # Add the score to the cumulative score for the selected group
        cumulative_scores[group_name] += score
        counts[group_name] += 1

    # Normalize the scores based on how many times each group was sampled
    for group_name in cumulative_scores.keys():
        if counts[group_name] > 0:
            cumulative_scores[group_name] /= counts[group_name]
    
    return cumulative_scores

# Example usage
groups = {
    "Group 1": ["apple", "apelp", "palle", "appel"],
    "Group 2": ["dog", "god", "odg", "gdo"],
    "Group 3": ["car", "arc", "rac", "acr"],
}

target_string = "appel"

cumulative_scores = monte_carlo_cumulative_match(target_string, groups, trials=10000)

# Print the cumulative scores for each group
for group_name, score in cumulative_scores.items():
    print(f"{group_name}: Average similarity score = {score:.4f}")


import random
import Levenshtein

def random_mutation(string, mutation_rate=0.1):
    """Randomly mutates a string by shuffling or making edits."""
    mutated = list(string)
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly delete or replace a character
            if random.random() < 0.5:
                # Replace with a random character
                mutated[i] = chr(random.randint(97, 122))  # Random lowercase letter
            else:
                # Delete a character
                mutated.pop(i)
    
    return ''.join(mutated)

def monte_carlo_string_matching(target_string, groups, trials=1000, mutation_rate=0.1):
    best_group = None
    best_score = float('inf')  # Start with a high score (lower is better)
    group_scores = {group_name: 0 for group_name in groups}  # Cumulative scores for each group

    for _ in range(trials):
        # Randomly mutate the target string
        mutated_target = random_mutation(target_string, mutation_rate)

        # Compare to all substrings in all groups
        for group_name, substrings in groups.items():
            for substring in substrings:
                # Randomly mutate the substring for variety
                mutated_substring = random_mutation(substring, mutation_rate)

                # Calculate Levenshtein distance (or any other metric)
                score = Levenshtein.distance(mutated_target, mutated_substring)
                
                # Normalize score by the length of the longer string to avoid bias
                normalized_score = score / max(len(mutated_target), len(mutated_substring))

                # Accumulate the score for the group
                group_scores[group_name] += normalized_score

                # Keep track of the best group found so far
                if normalized_score < best_score:
                    best_score = normalized_score
                    best_group = group_name

    # Return the group with the lowest cumulative score after all trials
    return min(group_scores, key=group_scores.get), group_scores

# Example usage
groups = {
    "Group 1": ["apple", "apelp", "pale"],
    "Group 2": ["dog", "god", "dogs", "gdo"],
    "Group 3": ["car", "arc", "racecar", "rac"],
}

target_string = "appel"

best_group, group_scores = monte_carlo_string_matching(target_string, groups)

print(f"The best match is in {best_group}")
print(f"Group scores: {group_scores}")



####### ahocorasick

import ahocorasick

def build_aho_corasick_automaton(groups):
    A = ahocorasick.Automaton()
    for group_name, substrings in groups.items():
        for substring in substrings:
            A.add_word(substring, (group_name, substring))
    A.make_automaton()
    return A

def find_matching_group_aho_corasick(target_string, automaton):
    matches = []
    for end_index, (group_name, substring) in automaton.iter(target_string):
        matches.append(group_name)
    return set(matches) if matches else "No matching group"

# Example usage
groups = {
    "Group 1": ["apple", "apelp", "palle", "appel"],
    "Group 2": ["dog", "god", "odg", "gdo"],
    "Group 3": ["car", "arc", "rac", "acr"],
}

automaton = build_aho_corasick_automaton(groups)
target_string = "appel"

matching_groups = find_matching_group_aho_corasick(target_string, automaton)
print(f"The target string matches group(s): {matching_groups}")


import ahocorasick

def build_aho_corasick_automaton_variable(groups):
    A = ahocorasick.Automaton()
    for group_name, substrings in groups.items():
        for substring in substrings:
            A.add_word(substring, (group_name, substring))
    A.make_automaton()
    return A

def find_matching_group_aho_corasick_variable(target_string, automaton):
    matches = []
    for end_index, (group_name, substring) in automaton.iter(target_string):
        matches.append(group_name)
    return set(matches) if matches else "No matching group"

# Example usage
groups = {
    "Group 1": ["apple", "apelp", "pale"],
    "Group 2": ["dog", "god", "dogs", "gdo"],
    "Group 3": ["car", "arc", "racecar", "rac"],
}

automaton = build_aho_corasick_automaton_variable(groups)
target_string = "appel"

matching_groups = find_matching_group_aho_corasick_variable(target_string, automaton)
print(f"The target string matches group(s): {matching_groups}")




####### cosine
from collections import Counter
import math

def cosine_similarity(str1, str2):
    vec1 = Counter(str1)
    vec2 = Counter(str2)
    
    # Calculate dot product and magnitude
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum([vec1[x] * vec2[x] for x in intersection])
    
    magnitude1 = math.sqrt(sum([val**2 for val in vec1.values()]))
    magnitude2 = math.sqrt(sum([val**2 for val in vec2.values()]))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def find_best_group_cosine(target_string, groups):
    best_group = None
    best_score = 0

    for group_name, substrings in groups.items():
        for substring in substrings:
            score = cosine_similarity(target_string, substring)
            if score > best_score:
                best_score = score
                best_group = group_name

    return best_group, best_score

# Example usage
groups = {
    "Group 1": ["apple", "apelp", "palle", "appel"],
    "Group 2": ["dog", "god", "odg", "gdo"],
    "Group 3": ["car", "arc", "rac", "acr"],
}

target_string = "appel"
best_group, best_score = find_best_group_cosine(target_string, groups)

print(f"The best match is in {best_group} with a cosine similarity score of {best_score:.4f}")

from collections import Counter
import math

def cosine_similarity_variable(str1, str2):
    vec1 = Counter(str1)
    vec2 = Counter(str2)
    
    # Calculate dot product and magnitude
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum([vec1[x] * vec2[x] for x in intersection])
    
    magnitude1 = math.sqrt(sum([val**2 for val in vec1.values()]))
    magnitude2 = math.sqrt(sum([val**2 for val in vec2.values()]))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def find_best_group_cosine_variable(target_string, groups):
    best_group = None
    best_score = 0

    for group_name, substrings in groups.items():
        for substring in substrings:
            score = cosine_similarity_variable(target_string, substring)
            if score > best_score:
                best_score = score
                best_group = group_name

    return best_group, best_score

# Example usage
groups = {
    "Group 1": ["apple", "apelp", "pale"],
    "Group 2": ["dog", "god", "dogs", "gdo"],
    "Group 3": ["car", "arc", "racecar", "rac"],
}

target_string = "appel"
best_group, best_score = find_best_group_cosine_variable(target_string, groups)

print(f"The best match is in {best_group} with a cosine similarity score of {best_score:.4f}")


