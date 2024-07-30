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


#-------

from itertools import product
from collections import defaultdict

def lcs_multiple(strings):
    if not strings:
        return []

    num_strings = len(strings)
    lengths = [len(s) for s in strings]

    # Create a multi-dimensional array to store lengths of LCS and sets of LCS substrings with counts
    dp = {}
    for indices in product(*(range(length + 1) for length in lengths)):
        dp[indices] = (0, defaultdict(int))

    # Build the dp array in bottom-up fashion
    found_common_substring = False
    for indices in product(*(range(length + 1) for length in lengths)):
        if all(index == 0 for index in indices):
            continue

        max_length = 0
        max_subseqs = defaultdict(int)
        for i in range(num_strings):
            if indices[i] > 0:
                prev_indices = indices[:i] + (indices[i] - 1,) + indices[i + 1:]
                if dp[prev_indices][0] > max_length:
                    max_length = dp[prev_indices][0]
                    max_subseqs = dp[prev_indices][1].copy()

                elif dp[prev_indices][0] == max_length:
                    for subseq, count in dp[prev_indices][1].items():
                        max_subseqs[subseq] += count

        current_chars = [strings[i][indices[i] - 1] for i in range(num_strings) if indices[i] > 0]
        if len(current_chars) == num_strings and all(char == current_chars[0] for char in current_chars):
            found_common_substring = True
            prev_indices = tuple(index - 1 for index in indices)
            new_subseqs = defaultdict(int)
            for subseq, count in dp[prev_indices][1].items() or {("", 1)}:
                new_subseqs[subseq + current_chars[0]] = count

            if dp[prev_indices][0] + 1 > max_length:
                max_length = dp[prev_indices][0] + 1
                max_subseqs = new_subseqs

            elif dp[prev_indices][0] + 1 == max_length:
                for subseq, count in new_subseqs.items():
                    max_subseqs[subseq] += count

        dp[indices] = (max_length, max_subseqs)

    # If no common substring is found
    if not found_common_substring:
        return []

    # The LCS with their counts is in dp[lengths]
    _, lcs_dict = dp[tuple(lengths)]
    return sorted(lcs_dict.items(), key=lambda x: (-x[1], x[0]))  # Sort by count (descending) and lexicographically

# Example usage
je = ['22222', 'aaaaa2222', 'c22']
lcs_results = lcs_multiple(je)
print("LCS results (substring, count):", lcs_results)

# Test with no common substring
je_no_common = ['abc', 'def', 'ghi']
lcs_results_no_common = lcs_multiple(je_no_common)
print("LCS results (no common substring):", lcs_results_no_common)

