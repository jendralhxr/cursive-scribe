import textdistance

# Strings to compare
str1 = "665-5405"  # ya of some sort
str2 = "567544-37-"  # ya of some sort

# Compute similarities/distances using different algorithms
print("Jaro:", textdistance.jaro.similarity(str1, str2))
print(
    "Jaro-Winkler:", textdistance.jaro_winkler.similarity(str1, str2)
)  # Same as strcmp95
print("STRCMP95:", textdistance.strcmp95.similarity(str1, str2))  # Same as strcmp95
print("Gotoh:", textdistance.gotoh.similarity(str1, str2))
print("Jaccard:", textdistance.jaccard.similarity(str1, str2))
print("Tanimoto:", textdistance.tanimoto.similarity(str1, str2))  # Alias of Jaccard
print("Sorensen:", textdistance.sorensen.similarity(str1, str2))
print(
    "Sorensen-Dice:", textdistance.sorensen_dice.similarity(str1, str2)
)  # Same as Dice
print("Dice:", textdistance.dice.similarity(str1, str2))  # Alias of Sorensen-Dice
print("Tversky:", textdistance.tversky.similarity(str1, str2))
print("Overlap:", textdistance.overlap.similarity(str1, str2))
print("Cosine:", textdistance.cosine.similarity(str1, str2))
print("Monge-Elkan:", textdistance.monge_elkan.similarity(str1, str2))

# just another LCS
print("Bag:", textdistance.bag.similarity(str1, str2))


str1 = "665-5405"  # ya of some sort
str2 = "567544-37-"  # ya of some sort


def myjaro(s1, s2):
    prefix_weight: float = 0.1
    s1_len = len(s1)
    s2_len = len(s2)

    if not s1_len or not s2_len:
        result = 0
        # return 0.0

    min_len = min(s1_len, s2_len)
    search_range = max(s1_len, s2_len)
    search_range = (search_range // 2) - 1
    if search_range < 0:
        search_range = 0

    s1_flags = [False] * s1_len
    s2_flags = [False] * s2_len

    # looking only within search range, count & flag matched pairs
    common_chars = 0
    almost_common_chars = 0
    for i, s1_ch in enumerate(s1):
        low = max(0, i - search_range)
        hi = min(i + search_range, s2_len - 1)
        for j in range(low, hi + 1):
            # print(f"eval s1[{i}]:{s1[i]} s2[{j}]:{s2[j]}")
            if not s2_flags[j] and s2[j] == s1_ch:
                s1_flags[i] = s2_flags[j] = True
                common_chars += 1
                # print(f"com {i}-{j}: {s1_ch}")
                break
            if abs(ord(s1[i]) - ord(s2[j])) == 1 or abs(ord(s1[i]) - ord(s2[j])) == 7:
                almost_common_chars += 1
                # print(f"almostcom {i}-{j}: {s1[i]}")
                break

    if almost_common_chars and not common_chars:
        common_chars = 1e-12
    if not common_chars and not almost_common_chars:
        # return 0.0
        result = 0

    # count transpositions
    k = trans_count = 0
    for i, s1_f in enumerate(s1_flags):
        if s1_f:
            for j in range(k, s2_len):
                if s2_flags[j]:
                    k = j + 1
                    break
            if s1[i] != s2[j]:
                trans_count += 1
    trans_count //= 2

    # adjust for similarities in nonmatched characters
    weight = common_chars / s1_len + common_chars / s2_len
    weight += (common_chars - trans_count) / common_chars
    weight += (almost_common_chars) / (s1_len + s2_len) / 4
    weight /= 4

    # # stop to boost if strings are not similar
    # if not self.winklerize:
    #     return weight
    # if weight <= 0.7:
    #     return weight

    # winkler modification
    # adjust for up to first 4 chars in common
    j = min(min_len, 3)
    i = 0
    while i < j and s1[i] == s2[i]:
        i += 1
    if i:
        weight += i * prefix_weight * (1.0 - weight)

    # optionally adjust for long strings
    # after agreeing beginning chars, at least two or more must agree and
    # agreed characters must be > half of remaining characters
    # if not self.long_tolerance or min_len <= 4:
    #     return weight
    if common_chars <= i + 1 or 2 * common_chars < min_len + i:
        return weight
    tmp = (common_chars - i - 1) / (s1_len + s2_len - i * 2 + 2)
    weight += (1.0 - weight) * tmp
    return weight
