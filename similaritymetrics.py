import textdistance

# Strings to compare
str1 = "665-5405" # ya of some sort
str2 = "567544-37-" # ya of some sort

# Compute similarities/distances using different algorithms
print("Jaro:", textdistance.jaro.similarity(str1, str2))
print("Jaro-Winkler:", textdistance.jaro_winkler.similarity(str1, str2))  # Same as strcmp95
print("STRCMP95:", textdistance.strcmp95.similarity(str1, str2))  # Same as strcmp95
print("Gotoh:", textdistance.gotoh.similarity(str1, str2))
print("Jaccard:", textdistance.jaccard.similarity(str1, str2))
print("Tanimoto:", textdistance.tanimoto.similarity(str1, str2))  # Alias of Jaccard
print("Sorensen:", textdistance.sorensen.similarity(str1, str2))
print("Sorensen-Dice:", textdistance.sorensen_dice.similarity(str1, str2))  # Same as Dice
print("Dice:", textdistance.dice.similarity(str1, str2))  # Alias of Sorensen-Dice
print("Tversky:", textdistance.tversky.similarity(str1, str2))
print("Overlap:", textdistance.overlap.similarity(str1, str2))
print("Cosine:", textdistance.cosine.similarity(str1, str2))
print("Monge-Elkan:", textdistance.monge_elkan.similarity(str1, str2))

# just another LCS
print("Bag:", textdistance.bag.similarity(str1, str2))
