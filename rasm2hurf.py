import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random    
from scipy.signal import find_peaks
import seaborn as sns
from collections import defaultdict
from matplotlib.ticker import MultipleLocator, FuncFormatter
import re
from scipy.ndimage import gaussian_filter1d

# TODO later
# AND-OR graph network (based on tabulated fcs?)
# or perhaps a leaf node is a set of path with +/-1 variation

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)
NUM_HURF= 42
# the hurf lookup
hurf= [''] * NUM_HURF
hurf[1]= 'ا'
hurf[2]= 'ب'
hurf[3]= 'ت'
hurf[4]= 'ة'
hurf[5]= 'ث'
hurf[6]= 'ج'
hurf[7]= 'چ'
hurf[8]= 'ح'
hurf[9]= 'خ'
hurf[10]= 'د'
hurf[11]= 'ذ'
hurf[12]= 'ر'
hurf[13]= 'ز'
hurf[14]= 'س'
hurf[15]= 'ش'
hurf[16]= 'ص'
hurf[17]= 'ض'
hurf[18]= 'ط'
hurf[19]= 'ظ'
hurf[20]= 'ع'
hurf[21]= 'غ'
hurf[22]= 'ڠ'
hurf[23]= 'ف'
hurf[24]= 'ڤ'
hurf[25]= 'ق'
hurf[26]= 'ک'
hurf[27]= 'ݢ'
hurf[28]= 'ل'
hurf[29]= 'م'
hurf[30]= 'ن'
hurf[31]= 'و'
hurf[32]= 'ۏ'
hurf[33]= 'ه'
hurf[34]= 'ء'
hurf[35]= 'ي'
hurf[36]= 'ی'
hurf[37]= 'ڽ'
hurf[38]= 'ؤ'
hurf[39]= 'أ'
hurf[40]= 'ك'

NUM_CLUSTER= 16
clusters = [[] for _ in range(NUM_CLUSTER)]
clusters[0] = [1, 39, 36]          # ا, أ, ی
clusters[1] = [2, 3, 5, 30, 35, 37]  # ب, ت, ث, ن, ي, ڽ
clusters[2] = [6, 7, 8, 9]         # ج, چ, ح, خ
clusters[3] = [10, 11]             # د, ذ
clusters[4] = [12, 13]             # ر, ز
clusters[5] = [14, 15]             # س, ش
clusters[6] = [16, 17]             # ص, ض
clusters[7] = [18, 19]             # ط, ظ
clusters[8] = [20, 21, 22]         # ع, غ, ڠ
clusters[9] = [23, 24, 25]         # ف, ڤ, ق
clusters[10] = [26, 27, 40]        # ک, ݢ, ك
clusters[11] = [28]                # ل
clusters[12] = [29]                # م
clusters[13] = [31, 32, 38]        # و, ۏ, ؤ
clusters[14] = [33, 4]             # ه, ة
clusters[15] = [34]                # ء


def random_color():
    colormap = plt.get_cmap('nipy_spectral')  
    return colormap(random.random())

def map_huruf_to_val(huruf):
    try:
        return hurf.index(huruf)
    except ValueError:
        return None  # Return None for unmapped values

def is_single_char(value):
    return isinstance(value, str) and len(value) == 1


# Generate random data
np.random.seed(42)


fieldstring_merged= 'chaincode'
fieldstring= 'chaincode331'
fieldstring2= 'chaincode331b'

fieldhurf= 'huruf'
fieldval= 'val'

# annotated chaincodes and hurfs
source = pd.read_csv('perangjohorp1.csv')
source = source.reset_index(drop=True)
source = source[(source[fieldhurf] != "") & ((source[fieldstring] != "") | (source[fieldstring2] != ""))]
# some basic checks 
source = source[source[fieldhurf].notna() & (source[fieldstring].notna() | (source[fieldstring2].notna()))]
source[fieldval] = source[fieldhurf].apply(map_huruf_to_val)

# merging multiple chaincode columns
source_merged= pd.melt(source, id_vars=['huruf', 'val'], value_vars=[fieldstring, fieldstring2], 
                      var_name='chaincode_id', value_name=fieldstring_merged)
source_merged = source_merged.drop(columns=['chaincode_id'])
source= source_merged[source_merged[fieldstring_merged].notna() & source_merged[fieldval].notna() ]

#source['is_valid'] = source['huruf'].apply(is_single_char)
# source['chaincode'].str.len().mean()
# source['chaincode'].str.len().min()
# source['chaincode'].str.len().max()

#random_strings=pd.concat([source['2bfs'], source['2alpha-bfsdfs']])
#random_labels=pd.concat([source['label'], source['label']])
#random_strings=pd.concat([source['chaincode']])
#random_hurf=pd.concat([source['val']])
#random_labels=pd.concat([source['val']])

plt.figure(dpi=300)
plt.rcParams.update({
    'figure.dpi': 300,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'grid.color': 'gray',
    'grid.alpha': 0.3
})

# First plot - Histogram of character lengths
char_lengths = source[fieldstring_merged].apply(len)
plt.figure()  # Create a new figure
plt.hist(char_lengths, bins=range(min(char_lengths), max(char_lengths) + 1, 1), edgecolor='black')
plt.xlabel('Chaincode length')
plt.ylabel('Appearance')
plt.title("Character Length Distribution")
plt.xticks(range(0, max(char_lengths) + 1, 5))
plt.grid(True)

quartiles = char_lengths.quantile([0.25, 0.5, 0.75])
hist_charlen_val, hist_charlen_bin = np.histogram(char_lengths, \
    bins = np.arange(min(char_lengths), max(char_lengths)))
most_common_length= hist_charlen_bin[np.argmax(hist_charlen_val)]
    
# Parameters
LENGTH_MIN= 2
LENGTH_MAX = int(LENGTH_MIN*pow(PHI,5))  # Max length of the strings
NUM_CLASSES = NUM_HURF  # Number of classes
SUBSTROKE_MIN_LENGTH= int(LENGTH_MIN*PHI)

#### FCS stands for frequent common substring/subsequence/substroke
# since there is preference for longest ones, it is now FLCS/LFCS

FCS_MAX_NUM= 18
FCS_APPEARANCE_MIN= 2

tokens = {f'{i}': [] for i in range(0, NUM_CLASSES)} # substring/substroke/subchain/subsequence
appearance = np.zeros(NUM_CLASSES, dtype=float) # hurf appearance

from itertools import product

def generate_permutations(s):
    options = [[str( (int(char) + delta) % 8)  for delta in (-1, 0, 1)] for char in s]
    permutations = [''.join(p) for p in product(*options)]
    return permutations

def update_rasm_score(hurf_class, rasm_seq, exact):
    if hurf_class in tokens:
        for subsequence in tokens[hurf_class]:
            # found exactly matching
            if subsequence['seq'] == rasm_seq and exact == True:
                subsequence['score'] += pow(PHI,len(rasm_seq)/LENGTH_MIN/PHI)
                subsequence['freq'] += 1
                return True  # early exit if already present
            elif subsequence['seq'] == rasm_seq and exact == False:
                subsequence['score'] += pow(PHI,len(rasm_seq)/LENGTH_MIN/pow(PHI,2))
                subsequence['freq'] += 1/PHI
                return False
            
        # add new seq if not already present
        if exact==True:
            tokens[hurf_class].append({'seq': rasm_seq, 'freq':1, 'score': pow(PHI,len(rasm_seq)/LENGTH_MIN/PHI)})


def parse_chaincode(input_string):
    result = []
    
    # assuming no transition points
    current_group = input_string[0]  # Start with the first character
    stroke_prev= True
    input_string_nohist= re.sub(f"[{re.escape('-+abcABCx')}]", '', input_string)
    for i in range(1, len(input_string_nohist)):
        diff= abs ( ord(input_string_nohist[i]) - ord(input_string_nohist[i-1]) )
        if diff==0 or diff==1 or diff==7:  # smooth stroke Freeman code
            straight_stroke= True 
        else:  # Different digit
            straight_stroke= False
        
        # contiguous smooth stroke
        if stroke_prev==straight_stroke:
            current_group += input_string_nohist[i]
        # changing radical
        elif stroke_prev!=straight_stroke: 
            if len(current_group) >= SUBSTROKE_MIN_LENGTH:  # Ensure group has at least 4 chars
                result.append(current_group)
                current_group = input_string_nohist[i]
                if straight_stroke== True:
                    current_group = input_string_nohist[i-1]+current_group
            else:  # new substroke
                current_group += input_string_nohist[i]
        stroke_prev= straight_stroke
    if len(current_group) >= SUBSTROKE_MIN_LENGTH:
        result.append(current_group)
    elif result:  # merge remaining characters
        result[-(SUBSTROKE_MIN_LENGTH-1):] += current_group

    # parse the chaincode based on transition nodes    
    substrs = re.split(r'([+\-abcABCx])', input_string)
    substrs = [substr for substr in substrs \
              if substr not in ['-', '+', 'a', 'b', 'c', 'A', 'B', 'C', 'x'] \
                  and len(substr)>=1]
    # 1-character substr is to be appended to previous substr also prepended to the next substr
    substrs_res= []
    for i in range(len(substrs)):
        if len(substrs[i]) == 1:  
            if substrs_res:  # Append to the previous string if result is not empty
                substrs_res[-1] += substrs[i]
            if i + 1 < len(substrs):  # Prepend to the next string if there is a next substr
                substrs[i + 1] = substrs[i] + substrs[i + 1]
            # 1-character substr (is not added do not add it to the result)
        else:
            substrs_res.append(substrs[i])
    result += substrs_res
    
    return result


def fcs_tabulate(val, string):
    #print(f"class{val} {string}")
    appearance[val] += 1
    
    if string=='66':
        print("walaa!")
    
    substrokes= parse_chaincode(string)
    #print(f" class{val} {substrokes}") 
    
    unique_substrings = set()  # Use a set to store unique substrings
    for substring in substrokes:
        #unique_substrings.add(substring)
        if len(substring) >= LENGTH_MIN and substring not in unique_substrings:
            # print(f"adding {substring} to {val}")
            unique_substrings.add(substring)
            
            # the read [original] substring
            update_rasm_score(str(val), substring, True)
            
            # permutated-modified substring
            if len(substring) <= most_common_length * PHI: 
                permutated_strings= generate_permutations(substring)
                for perm in permutated_strings:
                    update_rasm_score(str(val), perm, False)
            
    # sliding substring sampler, the old
    # for i in range(length):
    #     for j in range(i + 1, length + 1):
    #         substring = string[i:j]
    #         if len(substring) > 2 and substring not in unique_substrings:  # agar tidak overlap
    #             unique_substrings.add(substring)  
                

for i in range(0,source.shape[0]):
    #print(f"{i} {source[fieldstring][i]} {source[fieldval][i]}")
    fcs_tabulate(int(source.iloc[i][fieldval]), \
                 re.sub(f"[{re.escape('-+abcABCx')}]", '', source.iloc[i][fieldstring_merged]))

# some hurfs are not in Dejavu (U+1890, U+1743)
plt.figure(dpi=300)
plt.plot(appearance)
plt.xticks(ticks=range(len(hurf)), labels=hurf)
plt.savefig("/shm/hurfappearance.png", dpi=300)

top_fcs = {}
for n in range(len(tokens)):
    top_fcs[str(n)] = sorted(\
                        (item for item in tokens[str(n)] \
                         if item['freq'] > PHI and item['score'] > FCS_APPEARANCE_MIN*pow(PHI,2)),\
                        key=lambda x: x['score'], reverse=True)

# check for duplicates
seq_indices = defaultdict(list)
for key, entries in top_fcs.items():
    for i, entry in enumerate(entries):
        #seq_indices[entry['seq']].append(key)
        seq_indices[entry['seq']].append(hurf[int(key)])
duplicates_fcs = {seq: indices for seq, indices in seq_indices.items() if len(indices) > 1}

# removing the duplicates to make LFCSs in each class are more 'unique'
for hurf_class, entries in top_fcs.items():
    top_fcs[hurf_class] = [entry for entry in entries if entry['seq'] not in duplicates_fcs]

FCS_THINNING= char_lengths.mode()[0] # the mode of strings length, still feels inappropriate

lfcs = np.zeros((NUM_CLASSES, FCS_MAX_NUM)) # chaincode length
ffcs = np.zeros((NUM_CLASSES, FCS_MAX_NUM)) # frequency
sfcs = np.zeros((NUM_CLASSES, FCS_MAX_NUM)) # overall score
afcs = [["" for i in range(FCS_MAX_NUM)] for j in range(NUM_CLASSES)] # the substring 

for j in range(0, NUM_CLASSES): # the hurf
    if top_fcs[str(j)] is not None:
        for i in range(0, len(top_fcs[str(j)]) ): # the subsequences
            if i >= FCS_MAX_NUM:
                break
            # score_max= max(top_fcs[str(j)], key=lambda x: x['score'])['score']/appearance[j]
            # if top_fcs[str(j)][i]['score']/appearance[j] > score_max/pow(PHI,FCS_THINNING):
            # if top_fcs[str(j)][i]['score']/appearance[j] > 0:
            if top_fcs[str(j)][i]['score']/appearance[j] > 0:
                ffcs[j][i] = top_fcs[str(j)][i]['freq'] # apperance frequency of each substring
                sfcs[j][i] = top_fcs[str(j)][i]['score'] # length-dependent score of each substring
                afcs[j][i] = top_fcs[str(j)][i]['seq'] # the substring itself
                lfcs[j][i] = len(top_fcs[str(j)][i]['seq']) # length of each substring


# important to plot these graphs 
plt.figure(dpi=300)
sns.set_theme(rc={
    'figure.figsize': (16, 8),
    'font.family': ['Noto Naskh Arabic', 'Noto Sans'],
    'font.size': 6,  # Adjust font size if necessary
    'xtick.labelsize': 6,
    'ytick.labelsize': 6
})
#sns.heatmap(lfcs, cmap='nipy_spectral', annot=True, cbar=True, fmt='g', annot_kws={"size": 4})
#sns.heatmap(sfcs, cmap='nipy_spectral', annot=afcs, cbar=True, fmt='', annot_kws={"size": 4}, cbar_kws={"ticks": np.arange(0, 1.01, 0.05), "format": "%.2f"})
sns.heatmap(sfcs, cmap='nipy_spectral', annot=afcs, cbar=True, fmt='', annot_kws={"size": 4}, cbar_kws={"format": "%.2f"})
plt.yticks(ticks=range(NUM_CLASSES), labels=hurf, rotation=0, fontsize=6)
plt.xticks(fontsize=6, rotation=0)
#plt.title("FCS score: PHI^(len(subsequence)/2) / hurf-apperance")
plt.savefig("/shm/heatmapLCS.png", dpi=300)
# plt.xticks(ticks=range(len(y_labels)), labels=y_labels)
#ax = plt.gca()
#for tick in ax.get_yticklabels():
#    tick.set_y(tick.get_position()[1] + 400)  # Move tick labels down
#plt.show()

# FCS probability within a hurf
bin_edges = np.arange(sfcs.min(), sfcs.max() + 2)  # +2 to include the max value as a bin edge
counts, bins = np.histogram(sfcs.flatten(), bins=bin_edges)
#counts, bins = np.histogram(sfcs.flatten(), bins=FCS_MAX_NUM)
adjusted_counts = counts / np.count_nonzero(sfcs)
plt.figure(dpi=300)
plt.bar(bins[:-1], adjusted_counts, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel("Probability of subsequence appearance for each hurf")
plt.ylabel("Adjusted Probability of subsequence from the kitab")
plt.title("Subsequence Appearance Probability")
plt.grid(True)  # Enable grid
# the limit
# distribution_threshold = 1 - (1/PHI)
# plt.axvline(x=distribution_threshold, color='red', linestyle='--', linewidth=2)
# plt.annotate("1-(1/φ)",
#              xy=(distribution_threshold, 0),
#              xytext=(distribution_threshold + 0.05, 0.3),
#              fontsize=12, color='black')

# minimum length of token to be identified
#LENGTH_MIN= np.min(lfcs[lfcs != 0])

def myjaro(s1,s2):
    s1_len = len(s1)
    s2_len = len(s2)

    if not s1_len or not s2_len:
        #result= 0
        return 0.0
    if s1==s2:
        return 1.0

    search_range = max(s1_len, s2_len)
    search_range = (search_range // 2) - 1
    if search_range < 0:
        search_range = 0

    s1_flags = [False] * s1_len
    s2_flags = [False] * s2_len

    # looking only within search range, count & flag matched pairs
    common_chars = 0
    almost_common_chars= 0
    for i, s1_ch in enumerate(s1):
        low = max(0, i - search_range)
        hi = min(i + search_range, s2_len - 1)
        for j in range(low, hi + 1):
            #print(f"eval {low}-{hi} s1[{i}]:{s1[i]} s2[{j}]:{s2[j]}")
            if not s2_flags[j] and s2[j] == s1_ch:
                s1_flags[i] = s2_flags[j] = True
                common_chars += 1
                #print(f"com {i}-{j}: {s1_ch}")
                break
            if abs(ord(s1[i])-ord(s2[j]))==1 or abs(ord(s1[i])-ord(s2[j]))==4 or abs(ord(s1[i])-ord(s2[j]))==7:
                almost_common_chars += 1
                #print(f"almostcom {i}-{j}: {s1[i]}")
                break
            
    if almost_common_chars and not common_chars:
        common_chars=1e-12
    if not common_chars and not almost_common_chars:
        return 0.0
        #result=0

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
    #weight += (common_chars - pow(PHI,trans_count)) / common_chars
    weight += (common_chars - trans_count*PHI) / common_chars
    weight += (almost_common_chars) / (s1_len+s2_len) /4
    weight /= 3

    # # stop to boost if strings are not similar
    # if not self.winklerize:f
    #     return weight
    # if weight <= 0.7:
    #     return weight

    # winkler modification
    # adjust for up to first 'winkler_window' chars in common
    # winkler_window= 6
    # prefix_weight: float = 0.1
    # min_len = min(s1_len, s2_len)
    # j = min(min_len, winkler_window)
    # i = 0
    # while i < j and s1[i] == s2[i]:
    #     i += 1
    # if i:
    #     weight += i * prefix_weight * (1.0 - weight)

    # optionally adjust for long strings
    # after agreeing beginning chars, at least two or more must agree and
    # agreed characters must be > half of remaining characters
    # if not self.long_tolerance or min_len <= 4:
    #     return weight
    # if common_chars <= i + 1 or 2 * common_chars < min_len + i:
    #     return weight
    # tmp = (common_chars - i - 1) / (s1_len + s2_len - i * 2 + 2)
    # weight += (1.0 - weight) * tmp
    
    if weight<0.375001: # between '222' and '666'
        weight= 0.0
    return weight

MC_RETRY_MAX= 1e5

def draw_heatmap(data, xlabel, ylabel, title):
    plt.figure(dpi=300)
    sns.set_theme(rc={
        'font.family': ['Noto Naskh Arabic', 'Noto Sans', 'DejaVu Sans'],
        'font.size': 6,  # Adjust font size if necessary
        'xtick.labelsize': 6,
        'ytick.labelsize': 6
    })
    ax= sns.heatmap(data, cmap='nipy_spectral', annot=True, cbar=True, annot_kws={"size": 4})
    plt.yticks(ticks=range(NUM_CLASSES), labels=hurf, rotation=0, fontsize=6)
    plt.xticks(fontsize=6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for label in ax.get_yticklabels():
        label.set_verticalalignment('top')  # 'bottom' moves the labels slightly down

def draw_heatmap_slice(data, xlabel, ylabel, title):
    plt.figure(dpi=300)
    sns.set_theme(rc={
        'font.family': ['Noto Naskh Arabic', 'Noto Sans', 'DejaVu Sans'],
        'font.size': 6,  # Adjust font size if necessary
        'xtick.labelsize': 6,
        'ytick.labelsize': 6
    })
    # Slice data to skip columns 0 and 1
    data_to_plot = data[:, 2:]  # Skips the first two columns

    # Create the heatmap
    ax = sns.heatmap(data_to_plot, cmap='nipy_spectral', annot=True, cbar=True, annot_kws={"size": 4})
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(hurf, rotation=0, fontsize=6)
    ax.set_xticks(range(data_to_plot.shape[1]))
    ax.set_xticklabels(range(2, data.shape[1]), ha='center', fontsize=6)  # Explicitly center-align x-ticks

    # Adjust axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Adjust vertical alignment for y-tick labels
    for label in ax.get_yticklabels():
        label.set_verticalalignment('top')  # 'bottom' moves the labels slightly down

        
def draw_3d_surface(data, xlabel, ylabel, zlabel, title, elev=30, azim=45):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Define X and Y axes (assuming data is a 2D numpy array)
    x = np.arange(data.shape[1])  # Number of columns in the data
    y = np.arange(data.shape[0])  # Number of rows in the data
    X, Y = np.meshgrid(x, y)

    # Plot the surface
    surf = ax.plot_surface(X, Y, data, cmap='nipy_spectral', edgecolor='none')
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_zlabel(zlabel, fontsize=6)
    ax.set_title(title, fontsize=8)

    fig.colorbar(surf)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.view_init(elev=elev, azim=azim)
    plt.show()   


LENGTH_MAX= 15 # guess this is more based than 12 for LSTM
 
def reverseFreeman(s):
    result = []
    for char in s:
        if char.isdigit() and 0 <= int(char) <= 7:
            # If the character is a digit between 0 and 7, process it
            new_char = str((int(char) + 4) % 8)
            result.append(new_char)
        else:
            # If it's a non-numeric character, leave it unchanged or handle as needed
            result.append(char)
    return ''.join(result)

# if numpy 2
# from numpy.dtypes import StringDType



FACTOR_LENGTH= 1.00
VARIANCE_THRESHOLD= 5 # 1/5 of variance asymptote value

# same transition of diacritics connection can exist up to 2 count
def check_substroke(s):
    num_dia_top=0
    num_dia_bot=0
    num_hist=0
    num_slant=0
    num_branch=0
    before_branch=0
    for c in s:
        if c=='-':
            num_hist += 1
        elif c=='+':
            num_slant += 1
        elif c in 'abc':
            num_dia_bot += 1
        elif c in 'ABC':
            num_dia_top += 1
        elif c=='x': # jumping branch
            if before_branch>=LENGTH_MIN:    
                num_branch += 1
                before_branch=0
        else:
            before_branch +=1 # valid freeman vane
    if num_hist+num_slant==3 or num_dia_top==2 or num_dia_bot==2 or num_branch==2:
        return False
    else:
        return True
            
# chaincode
# chaincode= '455555-5544+43436645x5x5-554-44+321A03' # سوة
# chaincode= '66' # ا

def string2rasm(chaincode):
    rasm=''
    substrokes= re.findall(r'[^-+abcABCx]+[-+abcABCx]?', chaincode)
    
    # the MC search
    while len(substrokes)>1 or (len(substrokes)==1 and len(substrokes[0])>=LENGTH_MIN):
        # append substroke(s) to create minimum workable tee 
        hurf_best=''
        idx_cur= 0
        tee_clean= ''
        tee= ''
        while len(tee_clean) <LENGTH_MIN: # minimum tee to begin the search
            tee += substrokes[idx_cur]
            idx_cur += 1
            tee_clean = re.sub(f"[{re.escape('-+abcABCx')}]", '', tee)
        
        # mininmum length of tokens to be searched
        score_mc_acc = np.zeros((NUM_CLASSES, int(LENGTH_MIN*PHI)), dtype=float)
        # score_mc_mul = np.ones((NUM_CLASSES, int(LENGTH_MIN*PHI)), dtype=float)
        # score_mc_met = np.zeros((NUM_CLASSES, int(LENGTH_MIN*PHI)), dtype=float)
        # string_mc_met = np.full((NUM_CLASSES, int(LENGTH_MIN*PHI)), "", dtype='<U20')
        
        # MC search for valid sequence of substrokes, as long check_substroke() is still valid
        while True: 
            print(tee)
            tee_clean=re.sub(f"[{re.escape('-+abcABCx')}]", '', tee)
            mc_length_min= int( max(LENGTH_MIN, len(tee_clean)/PHI))
            
            # resize the result buffers
            temp= score_mc_acc
            score_mc_acc = np.zeros((NUM_CLASSES, int(len(tee_clean)*PHI+1)), dtype=float)
            score_mc_acc[:, 0:temp.shape[1]] = temp
            # temp= score_mc_mul
            # score_mc_mul = np.ones((NUM_CLASSES, int(len(tee_clean)*PHI+1)), dtype=float)
            # score_mc_mul[:, 0:temp.shape[1]] = temp
            # temp= score_mc_met
            # score_mc_met = np.zeros((NUM_CLASSES, int(len(tee_clean)*PHI+1)), dtype=float)
            # score_mc_met[:, 0:temp.shape[1]] = temp
            # temp= string_mc_met
            # string_mc_met = np.full((NUM_CLASSES, int(len(tee_clean)*PHI+1)), "", dtype='<U20')
            # string_mc_met[:, 0:temp.shape[1]] = temp
            
            for mc_retry in range(int(MC_RETRY_MAX)):
                mc_retry += 1
                mc_class= random.randint(1, len(top_fcs)-1)  # randomize the class
                fcs_lookup=''
                fcs_prob=1.00
                # check if top_fcs for the class is present, i.e. the hurf appears in the annotated dataset
                if len(top_fcs[str(mc_class)]) != 0:
                    mc_retry += 1
                    # generate lookup pattern to be long enough
                    while len(fcs_lookup)< len(tee)*PHI:
                        mc_index= random.randint(0, len(afcs[mc_class])-1) 
                        fcs_lookup += afcs[mc_class][mc_index]
                        fcs_prob *= sfcs[mc_class][mc_index]
                    
                    # compare tee against FCS string with length from len(tee)/PHI until len(tee)/PHI
                    for mc_length in range(mc_length_min, int(len(tee_clean)*PHI+1)):
                        # just adjust FACTOR_LENGTH if prefer for longer substrokes
                        score_tee1= myjaro( tee_clean, fcs_lookup[:mc_length]) * pow(FACTOR_LENGTH, mc_length)
                        score_tee2= myjaro( reverseFreeman(tee_clean), fcs_lookup[:mc_length]) * pow(FACTOR_LENGTH, mc_length)
                        score_tee= max(score_tee1, score_tee2) # some strokes can be written back to front due to branching
                        
                        # cumulative addition
                        score_mc_acc[int(mc_class)][mc_length] += score_tee
                        
                        # # cumulative product
                        # score_mc_mul[int(mc_class)][mc_length] *= score_tee
                        
                        # # metropolis
                        # if score_tee > score_mc_met[int(mc_class)][mc_length]:
                        #     string_mc_met[int(mc_class)][mc_length]= fcs_lookup
                        #     #score_mc[int(mc_class)][mc_length] = score_tee # absolute metropolis
                        #     score_mc_met[int(mc_class)][mc_length]= (score_tee + score_mc_met[int(mc_class)][mc_length]) /2 # incremental metropolis
        
            # include more substrokes to be evaluated
            if idx_cur>=len(substrokes):
                break
            else:
                tee += substrokes[idx_cur]
            
            if check_substroke(tee):
                idx_cur += 1
            else:
                break
        
        draw_heatmap(score_mc_acc, 'hurf character length', 'hurf class', 'cumulative add MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
                      +'\n'+tee)
        # draw MC search results
        # draw_heatmap(score_mc_met, 'hurf character length', 'hurf class', 'metropolis MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
        #               +'\n'+tee_fin)
        # score_mc_mul[score_mc_mul == 1.0] = 0.0
        # draw_heatmap(score_mc_mul, 'hurf character length', 'hurf class', 'cumulative product MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
        #               +'\n'+tee_fin)
        
        # pick the maximum for each length length in each hurf into the cluster
        score_mc_acc_cluster = np.zeros((NUM_CLUSTER, score_mc_acc.shape[1]), dtype=float)
        for m in range(score_mc_acc.shape[1]):
            for n in range(NUM_CLUSTER):
                score_mc_acc_cluster[n][m]= np.max(score_mc_acc[clusters[n], m])
        # plot the cluster
        plt.figure(dpi=300)
        sns.heatmap(score_mc_acc_cluster, cmap='nipy_spectral', annot=True, cbar=True, annot_kws={"size": 4})
        
        # identify best substrokes, hurf/class (cluster), and length
        #cluster_best= np.argmax(np.sum(score_mc_acc_cluster, axis=1))
        #cluster_best= np.argmax(np.mean(score_mc_acc_cluster, axis=1))
        cluster_best= np.argmax(np.max(score_mc_acc_cluster, axis=1))
        row_sums = [score_mc_acc[row, :].sum() for row in clusters[cluster_best]]
        class_best = clusters[cluster_best][np.argmax(row_sums)]
        hurf_best= hurf[class_best]
        
        # plot the best cluster/class
        plt.figure(dpi=300)
        plt.plot(score_mc_acc_cluster[cluster_best], label=f"cluster[{cluster_best}]", color="blue")
        for n in range(len(clusters[cluster_best])):
            plt.plot(score_mc_acc[clusters[cluster_best][n]], label=f"{hurf[clusters[cluster_best][n]]}", color=random_color(), linestyle="dashdot")
        plt.legend()
        
        # naive global peak
        # peak_index = np.argmax(score) # global peak can be misleading
        # slightly less naive first peak
        # peak_index = next(i for i in range(1, len(score) - 1) \
        #                   if score[i] > score[i - 1] and score[i] > score[i + 1])
        # smoothed peak
        score= score_mc_acc_cluster[cluster_best]
        score_smoothed= gaussian_filter1d(score, 1/PHI)
        peak_index= find_peaks(score_smoothed)[0][0] # grab the first peak
        
        valley_index = -1
        for i in range(peak_index, len(score) - 1):
            # print(f"{i} {(score[i-1]-score[i]):.2f} {score[i]:.2f} {(score[i]-score[i + 1]):.2f}")
            if score[i] < score[i - 1] \
                and (score[i] < score[i + 1] or (score[i]-score[i + 1])<score[i]/pow(PHI,9)):
                valley_index = i
                break
        if valley_index== -1:
            valley_index= len(score)
        # len_best = min(max(valley_index, len_max, len(tee_clean)), len(score))
        len_max = np.argmax(np.max(score_mc_acc[clusters[cluster_best], :], axis=0))
        len_best = min(max(valley_index, len_max), len(tee_clean), len(score))
        
        # recreate the tee
        idx_best= 0
        tee=''
        shortest_length = len(min(substrokes, key=len))
        # tee= substrokes[idx_best]
        # tee_clean= re.sub(f"[{re.escape('-+abcABCx')}]", '', tee)
        while idx_best < len(substrokes):
            tee += substrokes[idx_best]
            tee_clean= re.sub(f"[{re.escape('-+abcABCx')}]", '', tee)
            if len(tee_clean)>=len_best-shortest_length:
                break
            else:
                idx_best += 1
        tee_best= tee # substring representing a hurf
        
        # diacritics handling
        if hurf_best=='ا' or hurf_best=='أ':
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best:
                hurf_best= 'أ'
            else:
                hurf_best= 'ا'
        if hurf_best=='ب' or hurf_best=='ت' or hurf_best=='ث' or hurf_best=='ن' or hurf_best=='ي' or hurf_best=='ڽ' or hurf_best=='ی':
            if 'C' in tee_best:
                hurf_best= 'ث'
            elif 'b' in tee_best or 'c' in tee_best:
                hurf_best= 'ي'
            elif 'B' in tee_best:
                hurf_best= 'ت'
            elif 'A' in tee_best:
                hurf_best= 'ن'
            elif 'a' in tee_best:
                hurf_best= 'ب'
            else:
                hurf_best= 'ی'
                # rule for ending-ya (ي) could be lacking for more elaborate rayhani style
        if hurf_best=='ج' or hurf_best=='چ' or hurf_best=='ح' or hurf_best=='خ':
            if 'b' in tee_best or 'c' in tee_best:
                hurf_best= 'چ'
            elif 'A' in tee_best:
                hurf_best= 'خ'
            elif 'a' in tee_best:
                hurf_best= 'ج'
            else:
                hurf_best= 'ح'
        if hurf_best=='د' or hurf_best=='ذ' :
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ذ'
            else:
                hurf_best= 'د'
        if hurf_best=='ر' or hurf_best=='ز' :
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ز'
            else:
                hurf_best= 'ر'
        if hurf_best=='س' or hurf_best=='ش' :
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ش'
            else:
                hurf_best= 'س'
        if hurf_best=='ص' or hurf_best=='ض' :
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ض'
            else:
                hurf_best= 'ص'
        if hurf_best=='ع' or hurf_best=='غ' or hurf_best=='ڠ':
            if 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ڠ'
            elif 'A' in tee_best :
                hurf_best= 'غ'
            else:
                hurf_best= 'ع'
        if hurf_best=='ف' or hurf_best=='ڤ' or hurf_best=='ق':
            if 'C' in tee_best :
                hurf_best= 'ڤ'
            elif 'B' in tee_best:
                hurf_best= 'ق'
            elif 'A' in tee_best:
                hurf_best= 'ف'
            else:
                hurf_best= 'ف'
        if hurf_best=='ک' or hurf_best=='ݢ' or hurf_best=='ك' or hurf_best=='ل':
            if 'B' in tee_best or 'C' in tee_best:
                hurf_best= 'ك'
            if 'A' in tee_best or 'a' in tee_best or 'b' in tee_best: # some styles write the dot either on top or bottom
                hurf_best= 'ݢ'
        if hurf_best=='و' or hurf_best=='ۏ' or hurf_best=='ؤ':
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best:
                # hurf_best= 'ۏ'
                hurf_best= 'ؤ' # perang hohor usually use this style
            else:
                hurf_best= 'و'
        if hurf_best=='ه' or hurf_best=='ة':
            if 'A' in tee_best or 'B' in tee_best  or 'C' in tee_best :
                hurf_best= 'ة'
                

        # append to rasm        
        rasm+= hurf_best
        
        # remove searched substrokes
        substrokes= substrokes[idx_best+1:]
        
        remainder_stroke= ''.join(substrokes)
        # terminus hurfs
        # let's not terminate the rasm if any diacritics is still present
        # most likely due to the ه thing
        if (hurf_best=='د' or hurf_best=='ذ' or hurf_best=='ز' or hurf_best=='ر' or \
            hurf_best=='ۏ' or hurf_best=='و' or hurf_best=='ؤ' or \
            hurf_best=='ا' or hurf_best=='أ' or hurf_best=='ی' )\
            and len(remainder_stroke) <= LENGTH_MIN \
            and (not re.search(r'[abcABC]',remainder_stroke)):
            rasm += ' ' # end of rasm
            break
        
        if len(substrokes)==0:
            rasm += ' ' # end of rasm
            break
         
    return rasm



# from syair perahu
# remainder_stroke= '66676543535364667075444' # terlaLU with pruning
# remainder_stroke= '66670766454734453556707155535440' # terlaLU without pruning

# MC_jagokandang
def string2rasm_old(chaincode):
    remainder_stroke= chaincode
    rasm=''
    
    # MC search search
    while len(remainder_stroke)>=2 and remainder_stroke!='':
        len_mc_max= min(len(remainder_stroke), LENGTH_MAX)
        hurf_best=''
        len_mc_search= LENGTH_MIN+len_mc_max+1
        score_mc_acc = np.zeros((NUM_CLASSES, len_mc_search), dtype=float)
        score_mc_mul = np.ones((NUM_CLASSES, len_mc_search), dtype=float)
        score_mc_met = np.zeros((NUM_CLASSES, len_mc_search), dtype=float)
        string_mc_met = np.full((NUM_CLASSES, len_mc_search), "", dtype='<U20')
        
        for mc_retry in range(int(MC_RETRY_MAX)):
            #print(f"ret: {mc_retry}")
            fcs_lookup=''
            mc_class= random.randint(1, len(top_fcs)-1)  # may also compare to the whole string
            # skipping terminus hurf is chaincode is still long, but و could indeed be quite long
            # if len(remainder_stroke)>LENGTH_MIN*pow(PHI,LENGTH_MIN) and\
            #     (hurf[mc_class]=='ا' or hurf[mc_class]=='د' or hurf[mc_class]=='ذ' or hurf[mc_class]=='ر' or hurf[mc_class]=='ز' or hurf[mc_class]=='و')  :
            #     continue
            if len(top_fcs[str(mc_class)]) != 0:
                fcs_prob= 1
                
                for len_mc in range(LENGTH_MIN, len_mc_search):
                    while len(fcs_lookup)<=len_mc:
                        mc_index= random.randint(0, len(afcs[mc_class])-1) 
                        fcs_lookup += afcs[mc_class][mc_index]
                        fcs_prob *= sfcs[mc_class][mc_index]
                
                    if len(top_fcs[ str(mc_class) ]) != 0:
                        # similarity evaluation
                        score_tee1= myjaro( remainder_stroke[0:len_mc], fcs_lookup) * pow(FACTOR_LENGTH,len_mc) # (optionally)
                        score_tee2= myjaro( reverseFreeman(remainder_stroke[0:len_mc]), fcs_lookup) * pow(FACTOR_LENGTH,len_mc) # (optionally)
                        if len_mc <= int (LENGTH_MIN * pow(PHI,2)):
                            score_tee= max(score_tee1, score_tee2) # * fcs_prob
                        else:
                            score_tee = score_tee1 # * fcs_prob
                        
                        # cumulative addition
                        score_mc_acc[int(mc_class)][len_mc] += score_tee
                        
                        # cumulative product
                        score_mc_mul[int(mc_class)][len_mc] *= score_tee*PHI
                        
                        # metropolis
                        if score_tee > score_mc_met[int(mc_class)][len_mc]:
                            string_mc_met[int(mc_class)][len_mc]= fcs_lookup
                            #score_mc[int(mc_class)][len_mc] = score_tee
                            score_mc_met[int(mc_class)][len_mc]= (score_tee + score_mc_met[int(mc_class)][len_mc]) /2
        
        # plot the stop selection criteria
        # draw_heatmap(score_mc_met, 'hurf character length', 'hurf class', 'metropolis MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
        #               +'\n'+remainder_stroke)
        draw_heatmap(score_mc_acc, 'hurf character length', 'hurf class', 'cumulative add MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
                      +'\n'+remainder_stroke)
        score_mc_mul[score_mc_mul == 1.0] = 0.0
        draw_heatmap(score_mc_mul, 'hurf character length', 'hurf class', 'cumulative product MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
                      +'\n'+remainder_stroke)
        
        # optimum class selection
        row_sums = np.sum(score_mc_acc, axis=1)
        peaks= find_peaks(row_sums)[0]
        # row_sums_nonzero = [x for x in row_sums if x != 0]
        # peaks= find_peaks(row_sums, threshold=np.mean(row_sums_nonzero)/len_mc_max)[0] # I dunno why the threshold works not quite right atm
        lookupCS = [[ lookup for lookup in string_mc_met[peak]] for peak in peaks]
        tophurf = [[ max(myjaro(remainder_stroke, lookup), myjaro(reverseFreeman(remainder_stroke), lookup)) \
        for lookup in string_mc_met[peak]] for peak in peaks]
        row_sums = np.sum(tophurf, axis=1)
        class_best= peaks[np.argmax(row_sums)];
        
        max_row = tophurf[np.argmax(row_sums)]
        hurf_best= hurf[class_best]
        len_best= np.argmax(max_row); # minimum tentative estimate for hurf length
        
        # optimum stop-length selection
        if len_best <= LENGTH_MIN*PHI:
            len_best += LENGTH_MIN
        else:
            asymptote = np.mean(max_row[-int((len_mc_max-LENGTH_MIN)/PHI):]) # shall we do row value?
            divergence_val = np.where(np.abs(max_row - asymptote) > asymptote/pow(PHI,4))[0][-1]
            column_variances = np.var(score_mc_met, axis=0)
            column_variances /= np.max(column_variances) # scale max to 1
            asymptote_var = np.mean(column_variances[-int((len_mc_max-LENGTH_MIN)/PHI):]) # shall we do variance?
            divergence_var= np.where(np.abs(column_variances - asymptote_var) > asymptote_var/(len_mc_max-LENGTH_MIN)/PHI )[0][-1]
            len_best= min(divergence_var, divergence_val)
            
            plt.figure(dpi=300)
            fig, ax = plt.subplots()
            ax.plot(max_row, color='red', label='best-match hurf values')
            ax.plot(column_variances, color='blue', label='inter-hurf variance')
            ax.set_title(f'{remainder_stroke} ({hurf[class_best]})', fontsize=14)
            ax.set_xlabel('hurf character length', fontsize=12)
            ax.set_ylabel('normalized similarity score', fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))
            ax.legend()
            ax.axvline(x=len_best, color='gray', linestyle='--', linewidth=1, label='divergence point')
            ax.annotate(f"optimum length is {len_best}", 
                xy=(len_best, ax.get_ylim()[1]), 
                xytext=(len_best + 0.5, ax.get_ylim()[1] * 0.3),
                #arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=12, color='black')
        if len_best > len(remainder_stroke):
            len_best= len(remainder_stroke)
        tee_best= remainder_stroke[:len_best]
        
        
        # diacritics selection
        if hurf_best=='ا' or hurf_best=='أ':
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best:
                hurf_best= 'أ'
            else:
                hurf_best= 'ا'
        if hurf_best=='ب' or hurf_best=='ت' or hurf_best=='ث' or hurf_best=='ن' or hurf_best=='ي' or hurf_best=='ڽ' or hurf_best=='ی':
            if 'A' in tee_best:
                hurf_best= 'ن'
            elif 'B' in tee_best:
                hurf_best= 'ت'
            elif 'C' in tee_best:
                hurf_best= 'ث'
            elif 'a' in tee_best:
                hurf_best= 'ب'
            elif 'b' in tee_best or 'c' in tee_best:
                hurf_best= 'ي'
            else:
                hurf_best= 'ی'
                # rule for ending-ya (ي) could be lacking for more elaborate rayhani style
        if hurf_best=='ج' or hurf_best=='چ' or hurf_best=='ح' or hurf_best=='خ':
            if 'A' in tee_best:
                hurf_best= 'خ'
            elif 'a' in tee_best:
                hurf_best= 'ج'
            elif 'b' in tee_best or 'c' in tee_best:
                hurf_best= 'چ'
            else:
                hurf_best= 'ح'
        if hurf_best=='د' or hurf_best=='ذ' :
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ذ'
            else:
                hurf_best= 'د'
        if hurf_best=='ر' or hurf_best=='ز' :
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ز'
            else:
                hurf_best= 'ر'
        if hurf_best=='س' or hurf_best=='ش' :
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ش'
            else:
                hurf_best= 'س'
        if hurf_best=='ص' or hurf_best=='ض' :
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ض'
            else:
                hurf_best= 'ص'
        if hurf_best=='ع' or hurf_best=='غ' or hurf_best=='ڠ':
            if 'A' in tee_best :
                hurf_best= 'غ'
            elif 'B' in tee_best or 'C' in tee_best  :
                hurf_best= 'ڠ'
            else:
                hurf_best= 'ع'
        if hurf_best=='ف' or hurf_best=='ڤ' or hurf_best=='ق':
            if 'A' in tee_best:
                hurf_best= 'ف'
            elif 'B' in tee_best:
                hurf_best= 'ق'
            elif 'C' in tee_best :
                hurf_best= 'ڤ'
            else:
                hurf_best= 'ف'
        if hurf_best=='ک' or hurf_best=='ݢ' or hurf_best=='ك' or hurf_best=='ل':
            if 'B' in tee_best or 'C' in tee_best:
                hurf_best= 'ك'
            if 'A' in tee_best or 'a' in tee_best or 'b' in tee_best: # some styles write the dot either on top or bottom
                hurf_best= 'ݢ'
        if hurf_best=='و' or hurf_best=='ۏ':
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best:
                hurf_best= 'ۏ'
            else:
                hurf_best= 'و'

        # append to rasm        
        rasm+= hurf_best

        # terminus hurfs            
        if (hurf_best=='د' or hurf_best=='ذ' or hurf_best=='ز' or hurf_best=='ۏ')\
            and len(remainder_stroke)-len_best < SUBSTROKE_MIN_LENGTH*PHI:
           # hurf_best=='ا' or hurf_best=='و ' 
           # wawu (و) are often connected to ه or ة, alif is merged along with ل
           remainder_stroke=''
        elif (hurf_best=='ا' or hurf_best=='و' or hurf_best=='ر' ) and len(remainder_stroke)-len_best > SUBSTROKE_MIN_LENGTH:
            rasm += ' ' # inter-rasm space
        else:
            remainder_stroke= remainder_stroke[len_best:]
        
        if remainder_stroke=='' or len(remainder_stroke)<2:
            break
        
    # full rasm (part of word) from a chaincode
    return(rasm)

import sys
file_path = sys.argv[1]

with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        #print(line, end='')  # The 'end' argument avoids adding extra newlines
        print(f"{string2rasm(line)} ")