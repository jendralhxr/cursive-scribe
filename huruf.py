import networkx as nx
import random

hurf = nx.Graph()

hurf.add_node(0, label=' ')
hurf.add_node(1, label='ا')
hurf.add_node(2, label='ب')
hurf.add_node(3, label='ت')
hurf.add_node(4, label='ة')
hurf.add_node(5, label='ث')
hurf.add_node(6, label='ج')
hurf.add_node(7, label='چ')
hurf.add_node(8, label='ح')
hurf.add_node(9, label='خ')
hurf.add_node(10, label='د')
hurf.add_node(11, label='ذ')
hurf.add_node(12, label='ر')
hurf.add_node(13, label='ز')
hurf.add_node(14, label='س')
hurf.add_node(15, label='ش')
hurf.add_node(16, label='ص')
hurf.add_node(17, label='ض')
hurf.add_node(18, label='ط')
hurf.add_node(19, label='ظ')
hurf.add_node(20, label='ع')
hurf.add_node(21, label='غ')
hurf.add_node(22, label='ڠ')
hurf.add_node(23, label='ف')
hurf.add_node(24, label='ڤ')
hurf.add_node(25, label='ق')
hurf.add_node(26, label='ک')
hurf.add_node(27, label='ݢ')
hurf.add_node(28, label='ل')
hurf.add_node(29, label='م')
hurf.add_node(30, label='ن')
hurf.add_node(31, label='و')
hurf.add_node(32, label='ۏ')
hurf.add_node(33, label='ه')
hurf.add_node(34, label='ء')
hurf.add_node(35, label='ي')
hurf.add_node(36, label='ی')
hurf.add_node(37, label='ڽ')

hurf.add_node(500, label=' ', desc='none')
hurf.add_node(511, label='﮲', desc='one dot over')
hurf.add_node(510, label='﮳', desc='one dot under')
hurf.add_node(521, label='﮴', desc='two dots over')
hurf.add_node(520, label='﮵', desc='two  dots under')
hurf.add_node(531, label='﮶', desc='three dots over')
hurf.add_node(530, label='﮷', desc='three dots under', )
hurf.add_node(55, label='ء', desc='hamza')


hurf.add_node(100, label='0')
hurf.add_node(101, label='1')
hurf.add_node(102, label='2')
hurf.add_node(103, label='3')
hurf.add_node(104, label='4')
hurf.add_node(105, label='5')
hurf.add_node(106, label='6')
hurf.add_node(107, label='7')

hurf.add_node(200, label='0')
hurf.add_node(201, label='1')
hurf.add_node(202, label='2')
hurf.add_node(203, label='3')
hurf.add_node(204, label='4')
hurf.add_node(205, label='5')
hurf.add_node(206, label='6')
hurf.add_node(207, label='7')

hurf.add_node(300, label='0')
hurf.add_node(301, label='1')
hurf.add_node(302, label='2')
hurf.add_node(303, label='3')
hurf.add_node(304, label='4')
hurf.add_node(305, label='5')
hurf.add_node(306, label='6')
hurf.add_node(307, label='7')

hurf.add_node(400, label='0')
hurf.add_node(401, label='1')
hurf.add_node(402, label='2')
hurf.add_node(403, label='3')
hurf.add_node(404, label='4')
hurf.add_node(405, label='5')
hurf.add_node(406, label='6')
hurf.add_node(407, label='7')

def add_to_hex_color(hex_color, r_add, g_add, b_add):
    r_hex = hex_color[1:3]
    g_hex = hex_color[3:5]
    b_hex = hex_color[5:7]
    r = int(r_hex, 16)
    g = int(g_hex, 16)
    b = int(b_hex, 16)
    r = r + r_add
    g = g + g_add
    b = b + b_add
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    r_hex = f"{r:02x}"
    g_hex = f"{g:02x}"
    b_hex = f"{b:02x}"
    new_hex_color = f"#{r_hex}{g_hex}{b_hex}"
    return new_hex_color

def add_hurf(G, pos, v1, v2, v3, v4, dia, hurf):
    # pos: 0 (isolated), 1 (initial), 2 (medial), 3 (final)
    # v1 through v4 is vane Freeman code
    # v1 is the (hopefully-)the first stroke, v4 is the last (left, usually)
    # dia is diacritics marks
    # hurf is the (supposed) target letter
    
    hue= random.randint(0, 160)
    if pos==0:
        base= '#000000'
        col= add_to_hex_color(base, hue, hue, hue)
    elif pos==1:
        base= '#000040'
        col= add_to_hex_color(base, 0, 0, hue)
    elif pos==2:
        base= '#004000'
        col= add_to_hex_color(base, 0, hue, 0)
    elif pos==3:
        base= '#400000'
        col= add_to_hex_color(base, hue, 0, 0)
        
    G.add_edge(100+v1, 200+v2, color=col)
    G.add_edge(200+v2, 300+v3, color=col)
    if v4!=0:
        G.add_edge(300+v3, 400+v4, color=col)
        G.add_edge(400+v4, dia, color=col)
    else:
        G.add_edge(300+v3, dia, color=col)
    G.add_edge(dia, hurf, color=col)

# isolated
add_hurf(hurf, 0, 6, 6, 6, 0, 500, 1) # 'ا'
add_hurf(hurf, 0, 2, 2, 2, 0, 500, 1) # 'ا'
add_hurf(hurf, 0, 6, 4, 2, 0, 510, 2) # 'ب'
add_hurf(hurf, 0, 6, 4, 2, 0, 521, 3) # 'ت'
add_hurf(hurf, 0, 7, 4, 1, 0, 521, 4) # 'ة'
add_hurf(hurf, 0, 4, 7, 1, 0, 521, 4) # 'ة'
add_hurf(hurf, 0, 6, 4, 2, 0, 531, 5) # 'ث'
add_hurf(hurf, 0, 4, 7, 4, 0, 510, 6) # 'ج'
add_hurf(hurf, 0, 4, 7, 4, 0, 530, 7) # 'چ'
add_hurf(hurf, 0, 4, 7, 4, 0, 500, 8) # 'ح'
add_hurf(hurf, 0, 4, 7, 4, 0, 511, 9) # 'خ'
add_hurf(hurf, 0, 7, 7, 4, 4, 500, 10) # 'د'
add_hurf(hurf, 0, 7, 7, 4, 4, 511, 11) # 'ذ'
add_hurf(hurf, 0, 6, 5, 5, 0, 500, 12) # 'ر'
add_hurf(hurf, 0, 6, 5, 5, 0, 511, 13) # 'ز'
add_hurf(hurf, 0, 5, 3, 5, 3, 500, 14) # 'س'
add_hurf(hurf, 0, 5, 3, 5, 3, 531, 15) # 'ش'
add_hurf(hurf, 0, 1, 6, 4, 5, 500, 16) # 'ص'
add_hurf(hurf, 0, 1, 6, 4, 5, 511, 17) # 'ض'
add_hurf(hurf, 0, 1, 6, 4, 2, 500, 18) # 'ط'
add_hurf(hurf, 0, 1, 6, 4, 2, 511, 19) # 'ظ'
add_hurf(hurf, 0, 4, 6, 0, 5, 500, 20) # 'ع'
add_hurf(hurf, 0, 4, 6, 0, 5, 511, 21) # 'غ'
add_hurf(hurf, 0, 4, 6, 0, 5, 531, 22) # 'ڠ'
add_hurf(hurf, 0, 1, 6, 4, 4, 511, 23) # 'ف'
add_hurf(hurf, 0, 1, 6, 4, 4, 531, 24) # 'ڤ'
add_hurf(hurf, 0, 1, 6, 4, 4, 521, 25) # 'ق'
add_hurf(hurf, 0, 4, 6, 7, 4, 500, 26) # 'ک'
add_hurf(hurf, 0, 5, 5, 7, 4, 500, 26) # 'ک'
add_hurf(hurf, 0, 4, 6, 7, 4, 511, 27) # 'ݢ'
add_hurf(hurf, 0, 5, 5, 7, 4, 511, 27) # 'ݢ'
add_hurf(hurf, 0, 6, 6, 4, 3, 500, 28) # 'ل'
add_hurf(hurf, 0, 7, 4, 6, 6, 500, 29) # 'م'
add_hurf(hurf, 0, 6, 4, 2, 0, 511, 30) # 'ن'
add_hurf(hurf, 0, 4, 1, 7, 5, 500, 31) # 'و'
add_hurf(hurf, 0, 4, 1, 7, 5, 511, 32) # 'ۏ'
add_hurf(hurf, 0, 4, 7, 1, 0, 500, 33) # 'ه'
add_hurf(hurf, 0, 7, 4, 1, 0, 500, 33) # 'ه'
add_hurf(hurf, 0, 5, 0, 5, 0, 500, 34) # 'ء'
add_hurf(hurf, 0, 1, 0, 5, 4, 520, 35) # 'ي'
add_hurf(hurf, 0, 1, 0, 5, 4, 500, 36) # 'ی'
add_hurf(hurf, 0, 6, 4, 2, 0, 531, 37) # 'ڽ'

# initial
add_hurf(hurf, 1, 6, 4, 3, 4, 510, 2) # 'ب'
add_hurf(hurf, 1, 6, 4, 2, 0, 511, 30) # 'ن'
add_hurf(hurf, 1, 6, 4, 3, 4, 521, 3) # 'ت'
add_hurf(hurf, 1, 6, 4, 3, 4, 531, 5) # 'ث'
add_hurf(hurf, 1, 6, 4, 3, 4, 531, 37) # 'ڽ'
add_hurf(hurf, 1, 6, 4, 3, 4, 520, 35) # 'ي'
add_hurf(hurf, 1, 6, 4, 3, 4, 500, 36) # 'ی'
add_hurf(hurf, 1, 7, 4, 4, 0, 510, 6) # 'ج'
add_hurf(hurf, 1, 7, 4, 4, 0, 530, 7) # 'چ'
add_hurf(hurf, 1, 7, 4, 4, 0, 500, 8) # 'ح'
add_hurf(hurf, 1, 7, 4, 4, 0, 511, 9) # 'خ'
add_hurf(hurf, 1, 6, 4, 6, 4, 500, 14) # 'س'
add_hurf(hurf, 1, 6, 4, 6, 4, 531, 15) # 'ش'
add_hurf(hurf, 1, 5, 7, 4, 4, 500, 16) # 'ص'
add_hurf(hurf, 1, 5, 7, 4, 4, 511, 17) # 'ض'
add_hurf(hurf, 1, 5, 7, 4, 2, 500, 18) # 'ط'
add_hurf(hurf, 1, 5, 7, 4, 2, 511, 19) # 'ظ'
add_hurf(hurf, 1, 5, 7, 4, 6, 500, 18) # 'ط'
add_hurf(hurf, 1, 5, 7, 4, 6, 511, 19) # 'ظ'
add_hurf(hurf, 1, 1, 7, 4, 0, 500, 20) # 'ع'
add_hurf(hurf, 1, 1, 7, 4, 0, 511, 21) # 'غ'
add_hurf(hurf, 1, 1, 7, 4, 0, 531, 22) # 'ڠ'
add_hurf(hurf, 1, 1, 6, 4, 0, 511, 23) # 'ف'
add_hurf(hurf, 1, 1, 6, 4, 4, 531, 24) # 'ڤ'
add_hurf(hurf, 1, 1, 6, 4, 4, 521, 25) # 'ق'
add_hurf(hurf, 1, 4, 6, 7, 4, 500, 26) # 'ک'
add_hurf(hurf, 1, 5, 5, 7, 4, 500, 26) # 'ک'
add_hurf(hurf, 1, 4, 6, 7, 4, 511, 27) # 'ݢ'
add_hurf(hurf, 1, 5, 5, 7, 4, 511, 27) # 'ݢ'
add_hurf(hurf, 1, 6, 6, 4, 4, 500, 28) # 'ل'
add_hurf(hurf, 1, 7, 4, 4, 0, 500, 29) # 'م'
add_hurf(hurf, 1, 3, 5, 4, 0, 500, 29) # 'م'

# medial
add_hurf(hurf, 2, 2, 4, 3, 4, 510, 2) # 'ب'
add_hurf(hurf, 2, 2, 4, 2, 0, 511, 30) # 'ن'
add_hurf(hurf, 2, 2, 4, 3, 4, 521, 3) # 'ت'
add_hurf(hurf, 2, 2, 4, 3, 4, 531, 5) # 'ث'
add_hurf(hurf, 2, 2, 4, 3, 4, 531, 37) # 'ڽ'
add_hurf(hurf, 2, 2, 4, 3, 4, 520, 35) # 'ي'
add_hurf(hurf, 2, 2, 4, 3, 4, 500, 36) # 'ی'
add_hurf(hurf, 2, 3, 4, 4, 0, 510, 6) # 'ج'
add_hurf(hurf, 2, 3, 4, 4, 0, 530, 7) # 'چ'
add_hurf(hurf, 2, 3, 4, 4, 0, 500, 8) # 'ح'
add_hurf(hurf, 2, 3, 4, 4, 0, 511, 9) # 'خ'
add_hurf(hurf, 2, 3, 4, 3, 4, 500, 14) # 'س'
add_hurf(hurf, 2, 3, 4, 3, 4, 531, 15) # 'ش'
add_hurf(hurf, 2, 1, 1, 6, 4, 500, 16) # 'ص'
add_hurf(hurf, 2, 1, 1, 6, 4, 511, 17) # 'ض'
add_hurf(hurf, 2, 1, 1, 6, 2, 500, 18) # 'ط'
add_hurf(hurf, 2, 1, 1, 6, 2, 511, 19) # 'ظ'
add_hurf(hurf, 2, 1, 1, 6, 6, 500, 18) # 'ط'
add_hurf(hurf, 2, 1, 1, 6, 6, 511, 19) # 'ظ'
add_hurf(hurf, 2, 3, 0, 5, 4, 500, 20) # 'ع'
add_hurf(hurf, 2, 3, 0, 5, 4, 511, 21) # 'غ'
add_hurf(hurf, 2, 3, 0, 5, 4, 531, 22) # 'ڠ'
add_hurf(hurf, 2, 3, 4, 5, 4, 500, 20) # 'ع'
add_hurf(hurf, 2, 3, 4, 5, 4, 511, 21) # 'غ'
add_hurf(hurf, 2, 3, 4, 5, 4, 531, 22) # 'ڠ'
add_hurf(hurf, 2, 3, 4, 6, 4, 511, 23) # 'ف'
add_hurf(hurf, 2, 3, 4, 6, 4, 531, 24) # 'ڤ'
add_hurf(hurf, 2, 3, 4, 6, 4, 521, 25) # 'ق'
add_hurf(hurf, 2, 4, 2, 0, 4, 500, 26) # 'ک'
add_hurf(hurf, 2, 4, 2, 0, 4, 500, 26) # 'ک'
add_hurf(hurf, 2, 4, 3, 1, 4, 511, 27) # 'ݢ'
add_hurf(hurf, 2, 5, 3, 1, 4, 511, 27) # 'ݢ'
add_hurf(hurf, 2, 6, 6, 4, 4, 500, 28) # 'ل'
add_hurf(hurf, 2, 7, 4, 4, 4, 500, 29) # 'م'
add_hurf(hurf, 2, 3, 5, 4, 4, 500, 29) # 'م'