import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random    
from scipy.signal import find_peaks
import seaborn as sns
from collections import defaultdict
from matplotlib.ticker import MultipleLocator, FuncFormatter

# TODO-later
# AND-OR graph network (based on tabulated fcs?)
# or perhaps a leaf node is a set of path with +/-1 variation

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)

# the hurf lookup
hurf= [''] * 40
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


# Generate random data
np.random.seed(42)

# Parameters
LENGTH_MIN= 2
LENGTH_MAX = 16  # Max length of the strings
NUM_CLASSES = 40  # Number of classes

# data
source = pd.read_csv('syairperahu.csv')
source = source.reset_index(drop=True)

#random_strings=pd.concat([source['2bfs'], source['2alpha-bfsdfs']])
#random_labels=pd.concat([source['label'], source['label']])
random_strings=pd.concat([source['rasm']])
random_labels=pd.concat([source['val']])
#source['rasm'].str.len().mean()
#source['rasm'].str.len().min()
#source['rasm'].str.len().max()

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
char_lengths = source['rasm'].apply(len)
plt.figure()  # Create a new figure
plt.hist(char_lengths, bins=range(min(char_lengths), max(char_lengths) + 1), edgecolor='black')
plt.xlabel('chaincode length')
plt.ylabel('appearance')
plt.title("Character Length Distribution")
plt.grid(True)


quartiles = char_lengths.quantile([0.25, 0.5, 0.75])

#### FCS stands for frequent common substring/subsequence/substroke
# since there is preference for longest ones, it is now FLCS/LFCS

FCS_MAX_NUM= 18
FCS_APPEARANCE_MIN= 2

tokens = {f'{i}': [] for i in range(0, NUM_CLASSES)}
appearance = np.zeros(40, dtype=float) # hurf appearance

def update_rasm_score(hurf_class, rasm_seq):
    if hurf_class in tokens:
        for subsequence in tokens[hurf_class]:
            if subsequence['seq'] == rasm_seq:
                subsequence['score'] += pow(PHI,len(rasm_seq)/LENGTH_MIN)
                subsequence['freq'] += 1
                return True  # early exit if already present
        # add new seq if not already present
        tokens[hurf_class].append({'seq': rasm_seq, 'freq':1, 'score': pow(PHI,len(rasm_seq)/LENGTH_MIN)})

# TODO-later: pemotongan dengan proyeksi histogram untuk potong substroke
def parse_chaincode(input_string):
    SUBSTROKE_MIN_LENGTH= 4
    result = []
    current_group = input_string[0]  # Start with the first character
    stroke_prev= True
    
    for i in range(1, len(input_string)):
        diff= abs ( ord(input_string[i]) - ord(input_string[i-1]) )
        if diff==0 or diff==1 or diff==7:  # smooth stroke Freeman code
            straight_stroke= True 
        else:  # Different digit
            straight_stroke= False
        
        if stroke_prev==straight_stroke:
            current_group += input_string[i]
        elif stroke_prev!=straight_stroke:
            if len(current_group) >= SUBSTROKE_MIN_LENGTH:  # Ensure group has at least 4 chars
                result.append(current_group)
                current_group = input_string[i]
                if straight_stroke== True:
                    current_group = input_string[i-1]+current_group
            else:  # new substroke
                current_group += input_string[i]
        stroke_prev= straight_stroke
    
    if len(current_group) >= SUBSTROKE_MIN_LENGTH:
        result.append(current_group)
    elif result:  # merge remaining characters
        result[-(SUBSTROKE_MIN_LENGTH-1):] += current_group

    return result

def fcs_tabulate(val, string):
    #print(f"class{val} {string}")
    appearance[val] += 1
    length = len(string)
    
    substrokes= parse_chaincode(string)
    #print(f" class{val} {substrokes}") 
    
    unique_substrings = set()  # Use a set to store unique substrings
    for substring in substrokes:
        #unique_substrings.add(substring)
        if len(substring) > LENGTH_MIN and substring not in unique_substrings:
            #print(f"adding {substring} to {val}")
            unique_substrings.add(substring)
            update_rasm_score(str(val), substring)
            
    # sliding substing sampler, the old
    # for i in range(length):
    #     for j in range(i + 1, length + 1):
    #         substring = string[i:j]
    #         if len(substring) > 2 and substring not in unique_substrings:  # agar tidak overlap
    #             unique_substrings.add(substring)  
                


fieldstring= 'rasm'
fieldval= 'val'
for i in range(0,source.shape[0]):
    #print(f"{i} {source[fieldstring][i]} {source[fieldval][i]}")
    fcs_tabulate(int(source.iloc[i][fieldval]), str(source.iloc[i][fieldstring]).replace(" ", "").replace("+", "").replace("-", ""))

FCS_APPEARANCE_MIN= 0
top_fcs = {}
for hurf_class, rasm_seq in tokens.items():
    top_fcs[hurf_class] = sorted(\
        [x for x in rasm_seq if x['freq'] > FCS_APPEARANCE_MIN],
        key=lambda x: x['score'],
        reverse=True)

# check for duplicates
seq_indices = defaultdict(list)
for key, entries in top_fcs.items():
    for i, entry in enumerate(entries):
        #seq_indices[entry['seq']].append(key)
        seq_indices[entry['seq']].append(hurf[int(key)])
duplicates_fcs = {seq: indices for seq, indices in seq_indices.items() if len(indices) > 1}

# removing the duplicates
for hurf_class, entries in top_fcs.items():
    top_fcs[hurf_class] = [entry for entry in entries if entry['seq'] not in duplicates_fcs]

FCS_THINNING= char_lengths.mode()[0] # the mode of strings length, still feels inappropriate

lfcs = np.zeros((NUM_CLASSES, FCS_MAX_NUM))
ffcs = np.zeros((NUM_CLASSES, FCS_MAX_NUM))
sfcs = np.zeros((NUM_CLASSES, FCS_MAX_NUM))
afcs = [["" for i in range(FCS_MAX_NUM)] for j in range(NUM_CLASSES)]

for j in range(0, NUM_CLASSES): # the hurf
    if top_fcs[str(j)] is not None:
        for i in range(0, len(top_fcs[str(j)]) ): # the subsequences
            if i >= FCS_MAX_NUM:
                break
            score_max= max(top_fcs[str(j)], key=lambda x: x['score'])['score']/appearance[j]
            #if top_fcs[str(j)][i]['score']/appearance[j] > score_max/pow(PHI,FCS_THINNING):
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
plt.yticks(ticks=range(40), labels=hurf, rotation=0, fontsize=6)
plt.xticks(fontsize=6, rotation=0)
plt.title("FCS score: PHI^(len(subsequence)/2) / hurf-apperance")
#plt.savefig("/shm/heatmapLCS.png")
# plt.xticks(ticks=range(len(y_labels)), labels=y_labels)
#ax = plt.gca()
#for tick in ax.get_yticklabels():
#    tick.set_y(tick.get_position()[1] + 400)  # Move tick labels down
#plt.show()

# FCS probability within a hurf
counts, bins = np.histogram(sfcs.flatten(), bins=FCS_MAX_NUM)
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


def myjaro(s1,s2):
    prefix_weight: float = 0.1
    s1_len = len(s1)
    s2_len = len(s2)

    if not s1_len or not s2_len:
        #result= 0
        return 0.0
    if s1==s2:
        return 1.0

    min_len = min(s1_len, s2_len)
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
    # adjust for up to first 6 chars in common
    # j = min(min_len, 5)
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
    
    # TODO: discard low score, apply some threshold
    
    
    return weight

MC_RETRY_MAX= 1e4

def draw_heatmap(data, xlabel, ylabel, title):
    plt.figure(dpi=300)
    sns.set_theme(rc={
        'font.family': ['Noto Naskh Arabic', 'Noto Sans', 'DejaVu Sans'],
        'font.size': 6,  # Adjust font size if necessary
        'xtick.labelsize': 6,
        'ytick.labelsize': 6
    })
    ax= sns.heatmap(data, cmap='nipy_spectral', annot=True, cbar=True, annot_kws={"size": 4})
    plt.yticks(ticks=range(40), labels=hurf, rotation=0, fontsize=6)
    plt.xticks(fontsize=6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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

remainder_stroke= '66676543535364667075444' # terlaLU with pruning
remainder_stroke= '66670766454734453556707155535440' # terlaLU without pruning



def stringtorasm_MC_jagokandang(chaincode):
    remainder_stroke= chaincode
    rasm=''
    
    while len(remainder_stroke)>=2 and remainder_stroke!='':
        hurf_best=''
        len_mc_max= min(len(remainder_stroke), LENGTH_MAX)
        len_mc_search= LENGTH_MIN+len_mc_max+1
        score_mc = np.zeros((NUM_CLASSES, len_mc_search), dtype=float)
        score_mc_acc = np.zeros((NUM_CLASSES, len_mc_search), dtype=float)
        score_mc_mul = np.ones((NUM_CLASSES, len_mc_search), dtype=float)
        string_mc = np.full((NUM_CLASSES, len_mc_search), "", dtype='<U20')
        
        mc_retry= 0
        while(mc_retry < MC_RETRY_MAX):
            fcs_lookup=''
            mc_class= random.randint(1, len(top_fcs)-1)  # may also compare to the whole string
            if len(remainder_stroke)>LENGTH_MIN*pow(PHI,LENGTH_MIN) and\
                (hurf[mc_class]=='ا' or hurf[mc_class]=='د' or hurf[mc_class]=='ذ' or hurf[mc_class]=='ر' or hurf[mc_class]=='ز' or hurf[mc_class]=='و')  :
                continue
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
                        if score_tee > score_mc[int(mc_class)][len_mc]:
                            string_mc[int(mc_class)][len_mc]= fcs_lookup
                            #score_mc[int(mc_class)][len_mc] = score_tee
                            score_mc[int(mc_class)][len_mc]= (score_tee + score_mc[int(mc_class)][len_mc]) /2
                            
                mc_retry += 1 # can also be nested one down
        
        score_mc_mul[score_mc_mul == 1.0] = 0.0
        # plot the stop selection criteria
        # draw_heatmap(score_mc, 'hurf character length', 'hurf class', 'metropolis MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
        #               +'\n'+remainder_stroke)
        draw_heatmap(score_mc_acc, 'hurf character length', 'hurf class', 'cumulative add MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
                      +'\n'+remainder_stroke)
        # draw_heatmap(score_mc_mul, 'hurf character length', 'hurf class', 'cumulative product MC-myjaro '+str(int(MC_RETRY_MAX))+'/'+str(FACTOR_LENGTH)\
        #               +'\n'+remainder_stroke)
        
        # optimum class selection
        row_sums = np.sum(score_mc_acc, axis=1)
        row_sums_nonzero = [x for x in row_sums if x != 0]
        peaks= find_peaks(row_sums, threshold=np.mean(row_sums_nonzero)/len_mc_max)[0]
        lookupCS = [[ lookup for lookup in string_mc[peak]] for peak in peaks]
        tophurf = [[ max(myjaro(remainder_stroke, lookup), myjaro(reverseFreeman(remainder_stroke), lookup)) \
        for lookup in string_mc[peak]] for peak in peaks]
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
            column_variances = np.var(score_mc, axis=0)
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
            if 'A' in tee_best or 'a' in tee_best: # some styles write the dot either on top or bottom
                hurf_best= 'ݢ'
        if hurf_best=='و' or hurf_best=='ۏ':
            if 'A' in tee_best or 'B' in tee_best or 'C' in tee_best:
                hurf_best= 'ۏ'
            else:
                hurf_best= 'و'

        # append to rasm        
        rasm+= hurf_best

        # terminus hurfs            
        if hurf_best=='ا' or hurf_best=='د' or hurf_best=='ذ' or hurf_best=='ر' or hurf_best=='ز' or hurf_best=='و ' or hurf_best=='ۏ':
            remainder_stroke=''
        else:
            remainder_stroke= remainder_stroke[len_best:]
        
        if remainder_stroke=='' or len(remainder_stroke)<2:
            break
    return(rasm)

import sys
file_path = sys.argv[1]

with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        #print(line, end='')  # The 'end' argument avoids adding extra newlines
        print(f"{stringtorasm_MC_jagokandang(line)} ")