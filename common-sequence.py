str1 = "66676543535364667075444"  # terlaLU with pruning
str2 = "66670766454734453556707155535440"  # terlaLU without pruning


def lcs(str1, str2, m, n, track_m, track_n, count_track_pos):
    if len(str2) > len(str1):
        str_temp = str1
        str1 = str2
        str2 = str_temp
        temp = m
        m = n
        n = temp

    track_pos_m = []
    track_pos_n = []
    count_track_pos = 0
    if (
        m == 0
        or n == 0
        or len(track_pos_m) == len(str1)
        or len(track_pos_m) == len(str2)
        or len(track_pos_n) == len(str1)
        or len(track_pos_n) == len(str2)
    ):
        # print("return 0")
        return 0
    elif str1[m - 1] == str2[n - 1]:
        # print("track_m ",track_m)
        # print("track_n ",track_n)
        # print("str ",str1[ m-1], str2[ n-1], m-1, n-1)
        track_pos_m.append(-track_m)
        track_pos_n.append(-track_n)
        count_track_pos += 1
        track_m = track_m - 1
        track_n = track_n - 1
        # print("track_pos m ",track_pos_m)
        # print("track_pos n ",track_pos_m)
        return 1 + lcs(str1, str2, m - 1, n - 1, track_m, track_n, count_track_pos)
    else:
        # print("track_m ",track_m)
        # print("track_n ",track_n)
        # print("str- ",str1[ m-1], str2[ n-1], m-1, n-1)
        return max(
            lcs(str1, str2, m - 1, n, track_m - 1, track_n, count_track_pos),
            lcs(str1, str2, m, n - 1, track_m, track_n - 1, count_track_pos),
        )


# template, remainder_stroke) 5574 225633643644334141447320202273222231111
# str2 = "225633643644334141447320202273222231111"
# str1 = "225633643644334141447320202273222231111"
# input("Enter first string: ")

# str2 = "5574" #input("Enter second string: ")

# pencocokan lcs menggunakan and-or graph
# misal str2 = "4674" atau "5574"
# hasil lcs=2  atau 4 untuk str1 = "55577666546"  atau "55566654"


# dari travel-sales-problem.py
##k=10 list_vane [5, 5, 5, 7, 7, 6, 6, 6, 5, 4, 6]


# bagaimana string yg urutan terbalik ?
# misal dari sin dari travel-sales-problem.py

# str1 = "its"
# str2 = "winters"

# str1 = "55577666546"
# str2 = "5574"

# saya coba:
str2 = "66676543535364667075444"  # terlaLU with pruning
str1 = "66670766454734453556707155535440"  # terlaLU without pruning
# lcs_length = lcs(str1, str2, len(str1), len(str2),0,0)

# rekursinya berputar-putar m dan n bernilai 0, bergantian


# str1 = "55577666546"
# input("Enter first string: ")
# str2 = "4674" #kok sama dg 5574  lcs=2
# str2 = "6644" #5574" #input("Enter second string: ")
print("LEN ", len(str1), len(str2))
lcs_length = lcs(str1, str2, len(str1), len(str2), 0, 0, count_track_pos)
print("length of LCS is : {}".format(lcs_length))

# longest common sequence
# RS  225633643644334141447320202273222231111
# template,  222
# template, remainder_stroke) 5574 225633643644334141447320202273222231111
