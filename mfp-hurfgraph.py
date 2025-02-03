import networkx as nx
import random

hurf = nx.Graph()

# hurf.add_node(0, label=' ')
hurf.add_node(100, label="1→", pos=(0, 140))
hurf.add_node(101, label="1↗", pos=(0, 320))
hurf.add_node(102, label="1↑", pos=(0, 500))
hurf.add_node(103, label="1↖", pos=(0, 680))
hurf.add_node(104, label="1←", pos=(0, 860))
hurf.add_node(105, label="1↙", pos=(0, 1040))
hurf.add_node(106, label="1↓", pos=(0, 1220))
hurf.add_node(107, label="1↘", pos=(0, 1400))
hurf.add_node(200, label="2→", pos=(220, 140))
hurf.add_node(201, label="2↗", pos=(220, 320))
hurf.add_node(202, label="2↑", pos=(220, 500))
hurf.add_node(203, label="2↖", pos=(220, 680))
hurf.add_node(204, label="2←", pos=(220, 860))
hurf.add_node(205, label="2↙", pos=(220, 1040))
hurf.add_node(206, label="2↓", pos=(220, 1220))
hurf.add_node(207, label="2↘", pos=(220, 1400))
hurf.add_node(300, label="3→", pos=(440, 140))
hurf.add_node(301, label="3↗", pos=(440, 320))
hurf.add_node(302, label="3↑", pos=(440, 500))
hurf.add_node(303, label="3↖", pos=(440, 680))
hurf.add_node(304, label="3←", pos=(440, 860))
hurf.add_node(305, label="3↙", pos=(440, 1040))
hurf.add_node(306, label="3↓", pos=(440, 1220))
hurf.add_node(307, label="3↘", pos=(440, 1400))
hurf.add_node(400, label="4→", pos=(660, 140))
hurf.add_node(401, label="4↗", pos=(660, 320))
hurf.add_node(402, label="4↑", pos=(660, 500))
hurf.add_node(403, label="4↖", pos=(660, 680))
hurf.add_node(404, label="4←", pos=(660, 860))
hurf.add_node(405, label="4↙", pos=(660, 1040))
hurf.add_node(406, label="4↓", pos=(660, 1220))
hurf.add_node(407, label="4↘", pos=(660, 1400))
hurf.add_node(500, label=" ", pos=(880, 20))
hurf.add_node(511, label="1`", pos=(880, 240))
hurf.add_node(510, label="1-", pos=(880, 460))
hurf.add_node(521, label="2`", pos=(880, 680))
hurf.add_node(520, label="2-", pos=(880, 900))
hurf.add_node(531, label="3`", pos=(880, 1120))
hurf.add_node(530, label="3-", pos=(880, 1340))
hurf.add_node(55, label="ء", pos=(880, 1560))
hurf.add_node(1, label="ا", pos=(1100, 0))
hurf.add_node(2, label="ب", pos=(1100, 45))
hurf.add_node(3, label="ت", pos=(1100, 90))
hurf.add_node(4, label="ة", pos=(1100, 135))
hurf.add_node(5, label="ث", pos=(1100, 180))
hurf.add_node(6, label="ج", pos=(1100, 225))
hurf.add_node(7, label="چ", pos=(1100, 270))
hurf.add_node(8, label="ح", pos=(1100, 315))
hurf.add_node(9, label="خ", pos=(1100, 360))
hurf.add_node(10, label="د", pos=(1100, 405))
hurf.add_node(11, label="ذ", pos=(1100, 450))
hurf.add_node(12, label="ر", pos=(1100, 495))
hurf.add_node(13, label="ز", pos=(1100, 540))
hurf.add_node(14, label="س", pos=(1100, 585))
hurf.add_node(15, label="ش", pos=(1100, 630))
hurf.add_node(16, label="ص", pos=(1100, 675))
hurf.add_node(17, label="ض", pos=(1100, 720))
hurf.add_node(18, label="ط", pos=(1100, 765))
hurf.add_node(19, label="ظ", pos=(1100, 810))
hurf.add_node(20, label="ع", pos=(1100, 855))
hurf.add_node(21, label="غ", pos=(1100, 900))
hurf.add_node(22, label="ڠ", pos=(1100, 945))
hurf.add_node(23, label="ف", pos=(1100, 990))
hurf.add_node(24, label="ڤ", pos=(1100, 1035))
hurf.add_node(25, label="ق", pos=(1100, 1080))
hurf.add_node(26, label="ک", pos=(1100, 1125))
hurf.add_node(27, label="ݢ", pos=(1100, 1170))
hurf.add_node(28, label="ل", pos=(1100, 1215))
hurf.add_node(29, label="م", pos=(1100, 1260))
hurf.add_node(30, label="ن", pos=(1100, 1305))
hurf.add_node(31, label="و", pos=(1100, 1350))
hurf.add_node(32, label="ۏ", pos=(1100, 1395))
hurf.add_node(33, label="ه", pos=(1100, 1440))
hurf.add_node(34, label="ء", pos=(1100, 1485))
hurf.add_node(35, label="ي", pos=(1100, 1530))
hurf.add_node(36, label="ی", pos=(1100, 1575))
hurf.add_node(37, label="ڽ", pos=(1100, 1620))


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


def add_stroke(G, pos, v1, v2, v3, v4, dia, hurf):
    # pos: 0 (isolated), 1 (initial), 2 (medial), 3 (final)
    # v1 through v4 is vane Freeman code
    # v1 is the (hopefully-)the first stroke, v4 is the last (left, usually)
    # dia is diacritics marks
    # hurf is the (supposed) target letter

    hue = random.randint(0, 160)
    if pos == 0:
        base = "#000000"
        col = add_to_hex_color(base, hue, hue, hue)
    elif pos == 1:
        base = "#000040"
        col = add_to_hex_color(base, 0, 0, hue)
    elif pos == 2:
        base = "#004000"
        col = add_to_hex_color(base, 0, hue, 0)
    elif pos == 3:
        base = "#400000"
        col = add_to_hex_color(base, hue, 0, 0)

    G.add_edge(100 + v1, 200 + v2, color=col)
    G.add_edge(200 + v2, 300 + v3, color=col)
    if v4 != 9:
        G.add_edge(300 + v3, 400 + v4, color=col)
        G.add_edge(400 + v4, dia, color=col)
    else:
        G.add_edge(300 + v3, dia, color=col)
    G.add_edge(dia, hurf, color=col)


# isolated
add_stroke(hurf, 0, 6, 6, 6, 9, 500, 1)  # 'ا'
add_stroke(hurf, 0, 2, 2, 2, 9, 500, 1)  # 'ا'
add_stroke(hurf, 0, 6, 4, 2, 9, 510, 2)  # 'ب'
add_stroke(hurf, 0, 6, 4, 2, 9, 521, 3)  # 'ت'
add_stroke(hurf, 0, 7, 4, 1, 9, 521, 4)  # 'ة'
add_stroke(hurf, 0, 4, 7, 1, 9, 521, 4)  # 'ة'
add_stroke(hurf, 0, 6, 4, 2, 9, 531, 5)  # 'ث'
add_stroke(hurf, 0, 6, 4, 2, 9, 531, 37)  # 'ڽ'
add_stroke(hurf, 0, 4, 7, 4, 9, 510, 6)  # 'ج'
add_stroke(hurf, 0, 4, 7, 4, 9, 530, 7)  # 'چ'
add_stroke(hurf, 0, 4, 7, 4, 9, 500, 8)  # 'ح'
add_stroke(hurf, 0, 4, 7, 4, 9, 511, 9)  # 'خ'
add_stroke(hurf, 0, 0, 7, 4, 9, 510, 6)  # 'ج'
add_stroke(hurf, 0, 0, 7, 4, 9, 530, 7)  # 'چ'
add_stroke(hurf, 0, 0, 7, 4, 9, 500, 8)  # 'ح'
add_stroke(hurf, 0, 0, 7, 4, 9, 511, 9)  # 'خ'
add_stroke(hurf, 0, 7, 7, 4, 4, 500, 10)  # 'د'
add_stroke(hurf, 0, 7, 7, 4, 4, 511, 11)  # 'ذ'
add_stroke(hurf, 0, 6, 5, 5, 9, 500, 12)  # 'ر'
add_stroke(hurf, 0, 6, 5, 5, 9, 511, 13)  # 'ز'
add_stroke(hurf, 0, 5, 3, 5, 3, 500, 14)  # 'س'
add_stroke(hurf, 0, 5, 3, 5, 3, 531, 15)  # 'ش'
add_stroke(hurf, 0, 1, 6, 4, 5, 500, 16)  # 'ص'
add_stroke(hurf, 0, 1, 6, 4, 5, 511, 17)  # 'ض'
add_stroke(hurf, 0, 1, 6, 4, 2, 500, 18)  # 'ط'
add_stroke(hurf, 0, 1, 6, 4, 2, 511, 19)  # 'ظ'
add_stroke(hurf, 0, 4, 6, 0, 5, 500, 20)  # 'ع'
add_stroke(hurf, 0, 4, 6, 0, 5, 511, 21)  # 'غ'
add_stroke(hurf, 0, 4, 6, 0, 5, 531, 22)  # 'ڠ'
add_stroke(hurf, 0, 1, 6, 4, 4, 511, 23)  # 'ف'
add_stroke(hurf, 0, 1, 6, 4, 4, 531, 24)  # 'ڤ'
add_stroke(hurf, 0, 1, 6, 4, 4, 521, 25)  # 'ق'
add_stroke(hurf, 0, 4, 6, 7, 4, 55, 26)  # 'ک'
add_stroke(hurf, 0, 5, 5, 7, 4, 55, 26)  # 'ک'
add_stroke(hurf, 0, 4, 6, 7, 4, 511, 27)  # 'ݢ'
add_stroke(hurf, 0, 5, 5, 7, 4, 511, 27)  # 'ݢ'
add_stroke(hurf, 0, 6, 6, 4, 3, 500, 28)  # 'ل'
add_stroke(hurf, 0, 7, 4, 6, 6, 500, 29)  # 'م'
add_stroke(hurf, 0, 6, 4, 2, 9, 511, 30)  # 'ن'
add_stroke(hurf, 0, 4, 1, 7, 5, 500, 31)  # 'و'
add_stroke(hurf, 0, 4, 1, 7, 5, 511, 32)  # 'ۏ'
add_stroke(hurf, 0, 4, 7, 1, 9, 500, 33)  # 'ه'
add_stroke(hurf, 0, 7, 4, 1, 9, 500, 33)  # 'ه'
add_stroke(hurf, 0, 5, 0, 5, 9, 55, 34)  # 'ء'
add_stroke(hurf, 0, 1, 0, 5, 4, 520, 35)  # 'ي'
add_stroke(hurf, 0, 1, 0, 5, 4, 500, 36)  # 'ی'

# initial
add_stroke(hurf, 1, 6, 4, 3, 4, 510, 2)  # 'ب'
add_stroke(hurf, 1, 6, 4, 3, 4, 511, 30)  # 'ن'
add_stroke(hurf, 1, 6, 4, 3, 4, 521, 3)  # 'ت'
add_stroke(hurf, 1, 6, 4, 3, 4, 531, 5)  # 'ث'
add_stroke(hurf, 1, 6, 4, 3, 4, 531, 37)  # 'ڽ'
add_stroke(hurf, 1, 6, 4, 3, 4, 520, 35)  # 'ي'
add_stroke(hurf, 1, 6, 4, 3, 4, 500, 36)  # 'ی'
add_stroke(hurf, 1, 7, 4, 4, 9, 510, 6)  # 'ج'
add_stroke(hurf, 1, 7, 4, 4, 9, 530, 7)  # 'چ'
add_stroke(hurf, 1, 7, 4, 4, 9, 500, 8)  # 'ح'
add_stroke(hurf, 1, 7, 4, 4, 9, 511, 9)  # 'خ'
add_stroke(hurf, 0, 7, 4, 4, 9, 510, 6)  # 'ج'
add_stroke(hurf, 0, 7, 4, 4, 9, 530, 7)  # 'چ'
add_stroke(hurf, 0, 7, 4, 4, 9, 500, 8)  # 'ح'
add_stroke(hurf, 0, 7, 4, 4, 9, 511, 9)  # 'خ'
add_stroke(hurf, 1, 6, 4, 6, 4, 500, 14)  # 'س'
add_stroke(hurf, 1, 6, 4, 6, 4, 531, 15)  # 'ش'
add_stroke(hurf, 1, 5, 7, 4, 4, 500, 16)  # 'ص'
add_stroke(hurf, 1, 5, 7, 4, 4, 511, 17)  # 'ض'
add_stroke(hurf, 1, 5, 7, 4, 2, 500, 18)  # 'ط'
add_stroke(hurf, 1, 5, 7, 4, 2, 511, 19)  # 'ظ'
add_stroke(hurf, 1, 5, 7, 4, 6, 500, 18)  # 'ط'
add_stroke(hurf, 1, 5, 7, 4, 6, 511, 19)  # 'ظ'
add_stroke(hurf, 1, 1, 7, 4, 9, 500, 20)  # 'ع'
add_stroke(hurf, 1, 1, 7, 4, 9, 511, 21)  # 'غ'
add_stroke(hurf, 1, 1, 7, 4, 9, 531, 22)  # 'ڠ'
add_stroke(hurf, 1, 1, 6, 4, 9, 511, 23)  # 'ف'
add_stroke(hurf, 1, 1, 6, 4, 4, 531, 24)  # 'ڤ'
add_stroke(hurf, 1, 1, 6, 4, 4, 521, 25)  # 'ق'
add_stroke(hurf, 1, 4, 6, 7, 4, 500, 26)  # 'ک'
add_stroke(hurf, 1, 5, 5, 7, 4, 500, 26)  # 'ک'
add_stroke(hurf, 1, 4, 6, 7, 4, 511, 27)  # 'ݢ'
add_stroke(hurf, 1, 5, 5, 7, 4, 511, 27)  # 'ݢ'
add_stroke(hurf, 1, 6, 6, 4, 4, 500, 28)  # 'ل'
add_stroke(hurf, 1, 7, 4, 4, 4, 500, 29)  # 'م'
add_stroke(hurf, 1, 3, 5, 4, 4, 500, 29)  # 'م'

# medial
add_stroke(hurf, 2, 2, 4, 3, 4, 510, 2)  # 'ب'
add_stroke(hurf, 2, 2, 4, 3, 4, 511, 30)  # 'ن'
add_stroke(hurf, 2, 2, 4, 3, 4, 521, 3)  # 'ت'
add_stroke(hurf, 2, 2, 4, 3, 4, 531, 5)  # 'ث'
add_stroke(hurf, 2, 2, 4, 3, 4, 531, 37)  # 'ڽ'
add_stroke(hurf, 2, 2, 4, 3, 4, 520, 35)  # 'ي'
add_stroke(hurf, 2, 2, 4, 3, 4, 500, 36)  # 'ی'
add_stroke(hurf, 2, 3, 4, 4, 9, 510, 6)  # 'ج'
add_stroke(hurf, 2, 3, 4, 4, 9, 530, 7)  # 'چ'
add_stroke(hurf, 2, 3, 4, 4, 9, 500, 8)  # 'ح'
add_stroke(hurf, 2, 3, 4, 4, 9, 511, 9)  # 'خ'
add_stroke(hurf, 2, 3, 4, 3, 4, 500, 14)  # 'س'
add_stroke(hurf, 2, 3, 4, 3, 4, 531, 15)  # 'ش'
add_stroke(hurf, 2, 1, 1, 6, 4, 500, 16)  # 'ص'
add_stroke(hurf, 2, 1, 1, 6, 4, 511, 17)  # 'ض'
add_stroke(hurf, 2, 1, 1, 6, 2, 500, 18)  # 'ط'
add_stroke(hurf, 2, 1, 1, 6, 2, 511, 19)  # 'ظ'
add_stroke(hurf, 2, 1, 1, 6, 6, 500, 18)  # 'ط'
add_stroke(hurf, 2, 1, 1, 6, 6, 511, 19)  # 'ظ'
add_stroke(hurf, 2, 3, 0, 5, 4, 500, 20)  # 'ع'
add_stroke(hurf, 2, 3, 0, 5, 4, 511, 21)  # 'غ'
add_stroke(hurf, 2, 3, 0, 5, 4, 531, 22)  # 'ڠ'
add_stroke(hurf, 2, 3, 4, 5, 4, 500, 20)  # 'ع'
add_stroke(hurf, 2, 3, 4, 5, 4, 511, 21)  # 'غ'
add_stroke(hurf, 2, 3, 4, 5, 4, 531, 22)  # 'ڠ'
add_stroke(hurf, 2, 3, 4, 6, 4, 511, 23)  # 'ف'
add_stroke(hurf, 2, 3, 4, 6, 4, 531, 24)  # 'ڤ'
add_stroke(hurf, 2, 3, 4, 6, 4, 521, 25)  # 'ق'
add_stroke(hurf, 2, 4, 3, 1, 1, 500, 26)  # 'ک'
add_stroke(hurf, 2, 4, 2, 0, 0, 500, 26)  # 'ک'
add_stroke(hurf, 2, 4, 3, 1, 1, 511, 27)  # 'ݢ'
add_stroke(hurf, 2, 4, 2, 0, 0, 511, 27)  # 'ݢ'
add_stroke(hurf, 2, 6, 6, 4, 4, 500, 28)  # 'ل'
add_stroke(hurf, 2, 7, 4, 4, 4, 500, 29)  # 'م'
add_stroke(hurf, 2, 3, 5, 4, 4, 500, 29)  # 'م'

# final
add_stroke(hurf, 3, 4, 6, 6, 9, 500, 1)  # 'ا'
add_stroke(hurf, 3, 4, 2, 2, 9, 500, 1)  # 'ا'
add_stroke(hurf, 3, 6, 4, 3, 2, 510, 2)  # 'ب'
add_stroke(hurf, 3, 6, 4, 3, 2, 521, 3)  # 'ت'
add_stroke(hurf, 3, 6, 4, 3, 2, 511, 30)  # 'ن'
add_stroke(hurf, 3, 6, 4, 3, 2, 531, 5)  # 'ث'
add_stroke(hurf, 3, 6, 4, 3, 2, 531, 37)  # 'ڽ'
add_stroke(hurf, 3, 6, 4, 3, 1, 510, 2)  # 'ب'
add_stroke(hurf, 3, 6, 4, 3, 1, 521, 3)  # 'ت'
add_stroke(hurf, 3, 6, 4, 3, 1, 511, 30)  # 'ن'
add_stroke(hurf, 3, 6, 4, 3, 1, 531, 5)  # 'ث'
add_stroke(hurf, 3, 6, 4, 3, 1, 531, 37)  # 'ڽ'
add_stroke(hurf, 3, 2, 2, 5, 0, 521, 4)  # 'ة'
add_stroke(hurf, 3, 2, 4, 1, 0, 521, 4)  # 'ة'
add_stroke(hurf, 3, 2, 2, 5, 0, 500, 33)  # 'ه'
add_stroke(hurf, 3, 2, 4, 1, 0, 500, 33)  # 'ه'
add_stroke(hurf, 3, 2, 2, 5, 1, 521, 4)  # 'ة'
add_stroke(hurf, 3, 2, 4, 1, 1, 521, 4)  # 'ة'
add_stroke(hurf, 3, 2, 2, 5, 1, 500, 33)  # 'ه'
add_stroke(hurf, 3, 2, 4, 1, 1, 500, 33)  # 'ه'
add_stroke(hurf, 3, 3, 4, 6, 7, 510, 6)  # 'ج'
add_stroke(hurf, 3, 3, 4, 6, 7, 530, 7)  # 'چ'
add_stroke(hurf, 3, 3, 4, 6, 7, 500, 8)  # 'ح'
add_stroke(hurf, 3, 3, 4, 6, 7, 511, 9)  # 'خ'
add_stroke(hurf, 3, 3, 5, 0, 5, 510, 6)  # 'ج'
add_stroke(hurf, 3, 3, 5, 0, 5, 530, 7)  # 'چ'
add_stroke(hurf, 3, 3, 5, 0, 5, 500, 8)  # 'ح'
add_stroke(hurf, 3, 3, 5, 0, 5, 511, 9)  # 'خ'
add_stroke(hurf, 3, 3, 5, 0, 1, 510, 6)  # 'ج'
add_stroke(hurf, 3, 3, 5, 0, 1, 530, 7)  # 'چ'
add_stroke(hurf, 3, 3, 5, 0, 1, 500, 8)  # 'ح'
add_stroke(hurf, 3, 3, 5, 0, 1, 511, 9)  # 'خ'
add_stroke(hurf, 3, 3, 3, 4, 4, 500, 10)  # 'د'
add_stroke(hurf, 3, 3, 3, 4, 4, 511, 11)  # 'ذ'
add_stroke(hurf, 3, 3, 5, 5, 5, 500, 12)  # 'ر'
add_stroke(hurf, 3, 3, 5, 5, 5, 511, 13)  # 'ز'
add_stroke(hurf, 3, 3, 4, 3, 4, 500, 14)  # 'س'
add_stroke(hurf, 3, 3, 4, 3, 4, 531, 15)  # 'ش'
add_stroke(hurf, 3, 3, 5, 4, 5, 500, 16)  # 'ص'
add_stroke(hurf, 3, 3, 5, 4, 5, 511, 17)  # 'ض'
add_stroke(hurf, 3, 1, 5, 4, 2, 500, 18)  # 'ط'
add_stroke(hurf, 3, 1, 5, 4, 2, 511, 19)  # 'ظ'
add_stroke(hurf, 3, 4, 1, 4, 2, 500, 18)  # 'ط'
add_stroke(hurf, 3, 4, 1, 4, 2, 511, 19)  # 'ظ'
add_stroke(hurf, 3, 3, 0, 6, 7, 500, 20)  # 'ع'
add_stroke(hurf, 3, 3, 0, 6, 7, 511, 21)  # 'غ'
add_stroke(hurf, 3, 3, 0, 6, 7, 531, 22)  # 'ڠ'
add_stroke(hurf, 3, 1, 6, 4, 4, 511, 23)  # 'ف'
add_stroke(hurf, 3, 3, 0, 4, 3, 531, 24)  # 'ڤ'
add_stroke(hurf, 3, 3, 0, 4, 3, 521, 25)  # 'ق'
add_stroke(hurf, 3, 4, 4, 2, 2, 55, 26)  # 'ک'
add_stroke(hurf, 3, 4, 4, 6, 6, 55, 26)  # 'ک'
add_stroke(hurf, 3, 4, 4, 2, 2, 511, 27)  # 'ݢ'
add_stroke(hurf, 3, 4, 4, 6, 6, 511, 27)  # 'ݢ'
add_stroke(hurf, 3, 2, 2, 5, 4, 500, 28)  # 'ل'
add_stroke(hurf, 3, 5, 4, 6, 6, 500, 28)  # 'ل'
add_stroke(hurf, 3, 7, 4, 6, 6, 500, 29)  # 'م'
add_stroke(hurf, 3, 3, 5, 6, 6, 500, 29)  # 'م'
add_stroke(hurf, 3, 4, 1, 7, 5, 500, 31)  # 'و'
add_stroke(hurf, 3, 4, 1, 7, 5, 511, 32)  # 'ۏ'
add_stroke(hurf, 3, 5, 5, 4, 3, 520, 35)  # 'ي'
add_stroke(hurf, 3, 5, 5, 4, 3, 500, 36)  # 'ی'
add_stroke(hurf, 3, 5, 4, 4, 3, 520, 35)  # 'ي'
add_stroke(hurf, 3, 5, 4, 4, 3, 500, 36)  # 'ی'


def draw_graph(graph, pos):
    # edges
    if pos == None:
        pos = nx.spring_layout(graph)  # positions for all nodes

    labels = nx.get_node_attributes(graph, "label")
    colors = nx.get_edge_attributes(graph, "color").values()

    nx.draw(
        graph,
        pos,
        # nodes' param
        with_labels=True,
        labels=labels,
        node_color="orange",
        node_size=60,
        font_size=8,
        # edges' param
        edge_color=colors,
    )


# draw_graph(hurf, None)
# draw_graph(hurf, nx.spiral_layout(hurf))
draw_graph(hurf, nx.get_node_attributes(hurf, "pos"))


# --------------

stroke = []
stroke.append([0, 6, 6, 6, 9, 500, 1])
stroke.append([0, 2, 2, 2, 9, 500, 1])
stroke.append([0, 6, 4, 2, 9, 510, 2])
stroke.append([0, 6, 4, 2, 9, 521, 3])
stroke.append([0, 7, 4, 1, 9, 521, 4])
stroke.append([0, 4, 7, 1, 9, 521, 4])
stroke.append([0, 6, 4, 2, 9, 531, 5])
stroke.append([0, 6, 4, 2, 9, 531, 37])
stroke.append([0, 4, 7, 4, 9, 510, 6])
stroke.append([0, 4, 7, 4, 9, 530, 7])
stroke.append([0, 4, 7, 4, 9, 500, 8])
stroke.append([0, 4, 7, 4, 9, 511, 9])
stroke.append([0, 0, 7, 4, 9, 510, 6])
stroke.append([0, 0, 7, 4, 9, 530, 7])
stroke.append([0, 0, 7, 4, 9, 500, 8])
stroke.append([0, 0, 7, 4, 9, 511, 9])
stroke.append([0, 7, 7, 4, 4, 500, 10])
stroke.append([0, 7, 7, 4, 4, 511, 11])
stroke.append([0, 6, 5, 5, 9, 500, 12])
stroke.append([0, 6, 5, 5, 9, 511, 13])
stroke.append([0, 5, 3, 5, 3, 500, 14])
stroke.append([0, 5, 3, 5, 3, 531, 15])
stroke.append([0, 1, 6, 4, 5, 500, 16])
stroke.append([0, 1, 6, 4, 5, 511, 17])
stroke.append([0, 1, 6, 4, 2, 500, 18])
stroke.append([0, 1, 6, 4, 2, 511, 19])
stroke.append([0, 4, 6, 0, 5, 500, 20])
stroke.append([0, 4, 6, 0, 5, 511, 21])
stroke.append([0, 4, 6, 0, 5, 531, 22])
stroke.append([0, 1, 6, 4, 4, 511, 23])
stroke.append([0, 1, 6, 4, 4, 531, 24])
stroke.append([0, 1, 6, 4, 4, 521, 25])
stroke.append([0, 4, 6, 7, 4, 55, 26])
stroke.append([0, 5, 5, 7, 4, 55, 26])
stroke.append([0, 4, 6, 7, 4, 511, 27])
stroke.append([0, 5, 5, 7, 4, 511, 27])
stroke.append([0, 6, 6, 4, 3, 500, 28])
stroke.append([0, 7, 4, 6, 6, 500, 29])
stroke.append([0, 6, 4, 2, 9, 511, 30])
stroke.append([0, 4, 1, 7, 5, 500, 31])
stroke.append([0, 4, 1, 7, 5, 511, 32])
stroke.append([0, 4, 7, 1, 9, 500, 33])
stroke.append([0, 7, 4, 1, 9, 500, 33])
stroke.append([0, 5, 0, 5, 9, 55, 34])
stroke.append([0, 1, 0, 5, 4, 520, 35])
stroke.append([0, 1, 0, 5, 4, 500, 36])
stroke.append([1, 6, 4, 3, 4, 510, 2])
stroke.append([1, 6, 4, 3, 4, 511, 30])
stroke.append([1, 6, 4, 3, 4, 521, 3])
stroke.append([1, 6, 4, 3, 4, 531, 5])
stroke.append([1, 6, 4, 3, 4, 531, 37])
stroke.append([1, 6, 4, 3, 4, 520, 35])
stroke.append([1, 6, 4, 3, 4, 500, 36])
stroke.append([1, 7, 4, 4, 9, 510, 6])
stroke.append([1, 7, 4, 4, 9, 530, 7])
stroke.append([1, 7, 4, 4, 9, 500, 8])
stroke.append([1, 7, 4, 4, 9, 511, 9])
stroke.append([0, 7, 4, 4, 9, 510, 6])
stroke.append([0, 7, 4, 4, 9, 530, 7])
stroke.append([0, 7, 4, 4, 9, 500, 8])
stroke.append([0, 7, 4, 4, 9, 511, 9])
stroke.append([1, 6, 4, 6, 4, 500, 14])
stroke.append([1, 6, 4, 6, 4, 531, 15])
stroke.append([1, 5, 7, 4, 4, 500, 16])
stroke.append([1, 5, 7, 4, 4, 511, 17])
stroke.append([1, 5, 7, 4, 2, 500, 18])
stroke.append([1, 5, 7, 4, 2, 511, 19])
stroke.append([1, 5, 7, 4, 6, 500, 18])
stroke.append([1, 5, 7, 4, 6, 511, 19])
stroke.append([1, 1, 7, 4, 9, 500, 20])
stroke.append([1, 1, 7, 4, 9, 511, 21])
stroke.append([1, 1, 7, 4, 9, 531, 22])
stroke.append([1, 1, 6, 4, 9, 511, 23])
stroke.append([1, 1, 6, 4, 4, 531, 24])
stroke.append([1, 1, 6, 4, 4, 521, 25])
stroke.append([1, 4, 6, 7, 4, 500, 26])
stroke.append([1, 5, 5, 7, 4, 500, 26])
stroke.append([1, 4, 6, 7, 4, 511, 27])
stroke.append([1, 5, 5, 7, 4, 511, 27])
stroke.append([1, 6, 6, 4, 4, 500, 28])
stroke.append([1, 7, 4, 4, 4, 500, 29])
stroke.append([1, 3, 5, 4, 4, 500, 29])
stroke.append([2, 2, 4, 3, 4, 510, 2])
stroke.append([2, 2, 4, 3, 4, 511, 30])
stroke.append([2, 2, 4, 3, 4, 521, 3])
stroke.append([2, 2, 4, 3, 4, 531, 5])
stroke.append([2, 2, 4, 3, 4, 531, 37])
stroke.append([2, 2, 4, 3, 4, 520, 35])
stroke.append([2, 2, 4, 3, 4, 500, 36])
stroke.append([2, 3, 4, 4, 9, 510, 6])
stroke.append([2, 3, 4, 4, 9, 530, 7])
stroke.append([2, 3, 4, 4, 9, 500, 8])
stroke.append([2, 3, 4, 4, 9, 511, 9])
stroke.append([2, 3, 4, 3, 4, 500, 14])
stroke.append([2, 3, 4, 3, 4, 531, 15])
stroke.append([2, 1, 1, 6, 4, 500, 16])
stroke.append([2, 1, 1, 6, 4, 511, 17])
stroke.append([2, 1, 1, 6, 2, 500, 18])
stroke.append([2, 1, 1, 6, 2, 511, 19])
stroke.append([2, 1, 1, 6, 6, 500, 18])
stroke.append([2, 1, 1, 6, 6, 511, 19])
stroke.append([2, 3, 0, 5, 4, 500, 20])
stroke.append([2, 3, 0, 5, 4, 511, 21])
stroke.append([2, 3, 0, 5, 4, 531, 22])
stroke.append([2, 3, 4, 5, 4, 500, 20])
stroke.append([2, 3, 4, 5, 4, 511, 21])
stroke.append([2, 3, 4, 5, 4, 531, 22])
stroke.append([2, 3, 4, 6, 4, 511, 23])
stroke.append([2, 3, 4, 6, 4, 531, 24])
stroke.append([2, 3, 4, 6, 4, 521, 25])
stroke.append([2, 4, 3, 1, 1, 500, 26])
stroke.append([2, 4, 2, 0, 0, 500, 26])
stroke.append([2, 4, 3, 1, 1, 511, 27])
stroke.append([2, 4, 2, 0, 0, 511, 27])
stroke.append([2, 6, 6, 4, 4, 500, 28])
stroke.append([2, 7, 4, 4, 4, 500, 29])
stroke.append([2, 3, 5, 4, 4, 500, 29])
stroke.append([3, 4, 6, 6, 9, 500, 1])
stroke.append([3, 4, 2, 2, 9, 500, 1])
stroke.append([3, 6, 4, 3, 2, 510, 2])
stroke.append([3, 6, 4, 3, 2, 521, 3])
stroke.append([3, 6, 4, 3, 2, 511, 30])
stroke.append([3, 6, 4, 3, 2, 531, 5])
stroke.append([3, 6, 4, 3, 2, 531, 37])
stroke.append([3, 6, 4, 3, 1, 510, 2])
stroke.append([3, 6, 4, 3, 1, 521, 3])
stroke.append([3, 6, 4, 3, 1, 511, 30])
stroke.append([3, 6, 4, 3, 1, 531, 5])
stroke.append([3, 6, 4, 3, 1, 531, 37])
stroke.append([3, 2, 2, 5, 0, 521, 4])
stroke.append([3, 2, 4, 1, 0, 521, 4])
stroke.append([3, 2, 2, 5, 0, 500, 33])
stroke.append([3, 2, 4, 1, 0, 500, 33])
stroke.append([3, 2, 2, 5, 1, 521, 4])
stroke.append([3, 2, 4, 1, 1, 521, 4])
stroke.append([3, 2, 2, 5, 1, 500, 33])
stroke.append([3, 2, 4, 1, 1, 500, 33])
stroke.append([3, 3, 4, 6, 7, 510, 6])
stroke.append([3, 3, 4, 6, 7, 530, 7])
stroke.append([3, 3, 4, 6, 7, 500, 8])
stroke.append([3, 3, 4, 6, 7, 511, 9])
stroke.append([3, 3, 5, 0, 5, 510, 6])
stroke.append([3, 3, 5, 0, 5, 530, 7])
stroke.append([3, 3, 5, 0, 5, 500, 8])
stroke.append([3, 3, 5, 0, 5, 511, 9])
stroke.append([3, 3, 5, 0, 1, 510, 6])
stroke.append([3, 3, 5, 0, 1, 530, 7])
stroke.append([3, 3, 5, 0, 1, 500, 8])
stroke.append([3, 3, 5, 0, 1, 511, 9])
stroke.append([3, 3, 3, 4, 4, 500, 10])
stroke.append([3, 3, 3, 4, 4, 511, 11])
stroke.append([3, 3, 5, 5, 5, 500, 12])
stroke.append([3, 3, 5, 5, 5, 511, 13])
stroke.append([3, 3, 4, 3, 4, 500, 14])
stroke.append([3, 3, 4, 3, 4, 531, 15])
stroke.append([3, 3, 5, 4, 5, 500, 16])
stroke.append([3, 3, 5, 4, 5, 511, 17])
stroke.append([3, 1, 5, 4, 2, 500, 18])
stroke.append([3, 1, 5, 4, 2, 511, 19])
stroke.append([3, 4, 1, 4, 2, 500, 18])
stroke.append([3, 4, 1, 4, 2, 511, 19])
stroke.append([3, 3, 0, 6, 7, 500, 20])
stroke.append([3, 3, 0, 6, 7, 511, 21])
stroke.append([3, 3, 0, 6, 7, 531, 22])
stroke.append([3, 1, 6, 4, 4, 511, 23])
stroke.append([3, 3, 0, 4, 3, 531, 24])
stroke.append([3, 3, 0, 4, 3, 521, 25])
stroke.append([3, 4, 4, 2, 2, 55, 26])
stroke.append([3, 4, 4, 6, 6, 55, 26])
stroke.append([3, 4, 4, 2, 2, 511, 27])
stroke.append([3, 4, 4, 6, 6, 511, 27])
stroke.append([3, 2, 2, 5, 4, 500, 28])
stroke.append([3, 5, 4, 6, 6, 500, 28])
stroke.append([3, 7, 4, 6, 6, 500, 29])
stroke.append([3, 3, 5, 6, 6, 500, 29])
stroke.append([3, 4, 1, 7, 5, 500, 31])
stroke.append([3, 4, 1, 7, 5, 511, 32])
stroke.append([3, 5, 5, 4, 3, 520, 35])
stroke.append([3, 5, 5, 4, 3, 500, 36])
stroke.append([3, 5, 4, 4, 3, 520, 35])
stroke.append([3, 5, 4, 4, 3, 500, 36])

######
# isolated
hurf = nx.Graph()
hurf.add_node("222", label="ا")
hurf.add_node("642-", label="ب")
hurf.add_node("642+", label="ت")
hurf.add_node("741+", label="ة")
hurf.add_node("471+", label="ة")
hurf.add_node("642+", label="ث")
hurf.add_node("642+", label="ڽ")
hurf.add_node("474-", label="ج")
hurf.add_node("474-", label="چ")
hurf.add_node("474", label="ح")
hurf.add_node("474+", label="خ")
hurf.add_node("074-", label="ج")
hurf.add_node("074-", label="چ")
hurf.add_node("074", label="ح")
hurf.add_node("074+", label="خ")
hurf.add_node("7744", label="د")
hurf.add_node("7744+", label="ذ")
hurf.add_node("655", label="ر")
hurf.add_node("655+", label="ز")
hurf.add_node("5353", label="س")
hurf.add_node("5353+", label="ش")
hurf.add_node("645", label="ص")
hurf.add_node("645+", label="ض")
hurf.add_node("6422", label="ط")
hurf.add_node("6422+", label="ظ")
hurf.add_node("4605", label="ع")
hurf.add_node("4605+", label="غ")
hurf.add_node("4605+", label="ڠ")
hurf.add_node("644+", label="ف")
hurf.add_node("644+", label="ڤ")
hurf.add_node("644+", label="ق")
hurf.add_node("4674", label="ک")
hurf.add_node("5574", label="ک")
hurf.add_node("4674+", label="ݢ")
hurf.add_node("5574+", label="ݢ")
hurf.add_node("6644", label="ل")
hurf.add_node("7466", label="م")
hurf.add_node("642+", label="ن")
hurf.add_node("4175", label="و")
hurf.add_node("4175+", label="ۏ")
hurf.add_node("471", label="ه")
hurf.add_node("741", label="ه")
hurf.add_node("505", label="ء")
hurf.add_node("054-", label="ي")
hurf.add_node("0543", label="ی")

# initial
hurf.add_node("6434-", label="ب")
hurf.add_node("6434+", label="ن")
hurf.add_node("6434+", label="ت")
hurf.add_node("6434+", label="ث")
hurf.add_node("6434+", label="ڽ")
hurf.add_node("6434-", label="ي")
hurf.add_node("6434", label="ی")
hurf.add_node("744-", label="ج")
hurf.add_node("744-", label="چ")
hurf.add_node("744", label="ح")
hurf.add_node("744+", label="خ")
hurf.add_node("0744-", label="ج")
hurf.add_node("0744-", label="چ")
hurf.add_node("0744", label="ح")
hurf.add_node("0744+", label="خ")
hurf.add_node("6464", label="س")
hurf.add_node("6464+", label="ش")
hurf.add_node("5744", label="ص")
hurf.add_node("5744+", label="ض")
hurf.add_node("5742", label="ط")
hurf.add_node("5742+", label="ظ")
hurf.add_node("5746", label="ط")
hurf.add_node("5746", label="ظ")
hurf.add_node("174", label="ع")
hurf.add_node("174+", label="غ")
hurf.add_node("174+", label="ڠ")
hurf.add_node("164+", label="ف")
hurf.add_node("1644+", label="ڤ")
hurf.add_node("1644+", label="ق")
hurf.add_node("4674", label="ک")
hurf.add_node("5574", label="ک")
hurf.add_node("4674", label="ݢ")
hurf.add_node("5574", label="ݢ")
hurf.add_node("6644", label="ل")
hurf.add_node("7444", label="م")
hurf.add_node("3544", label="م")

# medial
hurf.add_node("2434-", label="ب")
hurf.add_node("2434+", label="ن")
hurf.add_node("2434+", label="ت")
hurf.add_node("2434+", label="ث")
hurf.add_node("2434+", label="ڽ")
hurf.add_node("2434-", label="ي")
hurf.add_node("2434", label="ی")
hurf.add_node("344-", label="ج")
hurf.add_node("344-", label="چ")
hurf.add_node("344", label="ح")
hurf.add_node("344+", label="خ")
hurf.add_node("3434", label="س")
hurf.add_node("3434+", label="ش")
hurf.add_node("1164", label="ص")
hurf.add_node("1164", label="ض")
hurf.add_node("11622", label="ط")
hurf.add_node("11622+", label="ظ")
hurf.add_node("1166", label="ط")
hurf.add_node("1166+", label="ظ")
hurf.add_node("30544", label="ع")
hurf.add_node("30544+", label="غ")
hurf.add_node("30544+", label="ڠ")
hurf.add_node("34544", label="ع")
hurf.add_node("3454+", label="غ")
hurf.add_node("3454+", label="ڠ")
hurf.add_node("3464+", label="ف")
hurf.add_node("3464+", label="ڤ")
hurf.add_node("3464+", label="ق")
hurf.add_node("4311", label="ک")
hurf.add_node("4200", label="ک")
hurf.add_node("4311", label="ݢ")
hurf.add_node("4200", label="ݢ")
hurf.add_node("6644", label="ل")
hurf.add_node("7444", label="م")
hurf.add_node("3544", label="م")

# final
hurf.add_node("466", label="ا")
hurf.add_node("422", label="ا")
hurf.add_node("6432-", label="ب")
hurf.add_node("6432+", label="ت")
hurf.add_node("6432+", label="ن")
hurf.add_node("6432+", label="ث")
hurf.add_node("6432+", label="ڽ")
hurf.add_node("6431-", label="ب")
hurf.add_node("6431+", label="ت")
hurf.add_node("6431+", label="ن")
hurf.add_node("6431+", label="ث")
hurf.add_node("6431+", label="ڽ")
hurf.add_node("2250+", label="ة")
hurf.add_node("2410+", label="ة")
hurf.add_node("2250", label="ه")
hurf.add_node("2410", label="ه")
hurf.add_node("2251+", label="ة")
hurf.add_node("2411+", label="ة")
hurf.add_node("2251", label="ه")
hurf.add_node("2411", label="ه")
hurf.add_node("3467-", label="ج")
hurf.add_node("3467-", label="چ")
hurf.add_node("3467", label="ح")
hurf.add_node("3467+", label="خ")
hurf.add_node("3505-", label="ج")
hurf.add_node("3505-", label="چ")
hurf.add_node("3505", label="ح")
hurf.add_node("3505+", label="خ")
hurf.add_node("3501-", label="ج")
hurf.add_node("3501-", label="چ")
hurf.add_node("3501", label="ح")
hurf.add_node("3501+", label="خ")
hurf.add_node("3344", label="د")
hurf.add_node("3344+", label="ذ")
hurf.add_node("3555", label="ر")
hurf.add_node("3555+", label="ز")
hurf.add_node("3434", label="س")
hurf.add_node("3434+", label="ش")
hurf.add_node("3545", label="ص")
hurf.add_node("3545+", label="ض")
hurf.add_node("1542", label="ط")
hurf.add_node("1542+", label="ظ")
hurf.add_node("41422", label="ط")
hurf.add_node("41422+", label="ظ")
hurf.add_node("3067", label="ع")
hurf.add_node("3067", label="غ")
hurf.add_node("3067+", label="ڠ")
hurf.add_node("1644+", label="ف")
hurf.add_node("3043+", label="ڤ")
hurf.add_node("3043+", label="ق")
hurf.add_node("4422", label="ک")
hurf.add_node("4466", label="ک")
hurf.add_node("4422+", label="ݢ")
hurf.add_node("4466+", label="ݢ")
hurf.add_node("2254", label="ل")
hurf.add_node("5466", label="ل")
hurf.add_node("7466", label="م")
hurf.add_node("3566", label="م")
hurf.add_node("4175", label="و")
hurf.add_node("4175+", label="ۏ")
hurf.add_node("5543-", label="ي")
hurf.add_node("5543", label="ی")
hurf.add_node("5443-", label="ي")
hurf.add_node("5443", label="ی")
