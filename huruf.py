import networkx as nx
import random

hurf = nx.Graph()

#hurf.add_node(0, label=' ')
hurf.add_node( 100, label='1→', pos=( 0, 100 ))
hurf.add_node( 101, label='1↗', pos=( 0, 280 ))
hurf.add_node( 102, label='1↑', pos=( 0, 460 ))
hurf.add_node( 103, label='1↖', pos=( 0, 640 ))
hurf.add_node( 104, label='1←', pos=( 0, 820 ))
hurf.add_node( 105, label='1↙', pos=( 0, 1000 ))
hurf.add_node( 106, label='1↓', pos=( 0, 1180 ))
hurf.add_node( 107, label='1↘', pos=( 0, 1360 ))
hurf.add_node( 200, label='2→', pos=( 220, 100 ))
hurf.add_node( 201, label='2↗', pos=( 220, 280 ))
hurf.add_node( 202, label='2↑', pos=( 220, 460 ))
hurf.add_node( 203, label='2↖', pos=( 220, 640 ))
hurf.add_node( 204, label='2←', pos=( 220, 820 ))
hurf.add_node( 205, label='2↙', pos=( 220, 1000 ))
hurf.add_node( 206, label='2↓', pos=( 220, 1180 ))
hurf.add_node( 207, label='2↘', pos=( 220, 1360 ))
hurf.add_node( 300, label='3→', pos=( 440, 100 ))
hurf.add_node( 301, label='3↗', pos=( 440, 280 ))
hurf.add_node( 302, label='3↑', pos=( 440, 460 ))
hurf.add_node( 303, label='3↖', pos=( 440, 640 ))
hurf.add_node( 304, label='3←', pos=( 440, 820 ))
hurf.add_node( 305, label='3↙', pos=( 440, 1000 ))
hurf.add_node( 306, label='3↓', pos=( 440, 1180 ))
hurf.add_node( 307, label='3↘', pos=( 440, 1360 ))
hurf.add_node( 400, label='4→', pos=( 660, 100 ))
hurf.add_node( 401, label='4↗', pos=( 660, 280 ))
hurf.add_node( 402, label='4↑', pos=( 660, 460 ))
hurf.add_node( 403, label='4↖', pos=( 660, 640 ))
hurf.add_node( 404, label='4←', pos=( 660, 820 ))
hurf.add_node( 405, label='4↙', pos=( 660, 1000 ))
hurf.add_node( 406, label='4↓', pos=( 660, 1180 ))
hurf.add_node( 407, label='4↘', pos=( 660, 1360 ))
hurf.add_node( 500, label=' ', pos=( 880, 20 ))
hurf.add_node( 511, label='﮲', pos=( 880, 240 ))
hurf.add_node( 510, label='﮳', pos=( 880, 460 ))
hurf.add_node( 521, label='﮴', pos=( 880, 680 ))
hurf.add_node( 520, label='﮵', pos=( 880, 900 ))
hurf.add_node( 531, label='﮶', pos=( 880, 1120 ))
hurf.add_node( 530, label='﮷', pos=( 880, 1340 ))
hurf.add_node( 55, label='ء', pos=( 880, 1560 ))
hurf.add_node( 1, label='ا', pos=( 1100, 0 ))
hurf.add_node( 2, label='ب', pos=( 1100, 45 ))
hurf.add_node( 3, label='ت', pos=( 1100, 90 ))
hurf.add_node( 4, label='ة', pos=( 1100, 135 ))
hurf.add_node( 5, label='ث', pos=( 1100, 180 ))
hurf.add_node( 6, label='ج', pos=( 1100, 225 ))
hurf.add_node( 7, label='چ', pos=( 1100, 270 ))
hurf.add_node( 8, label='ح', pos=( 1100, 315 ))
hurf.add_node( 9, label='خ', pos=( 1100, 360 ))
hurf.add_node( 10, label='د', pos=( 1100, 405 ))
hurf.add_node( 11, label='ذ', pos=( 1100, 450 ))
hurf.add_node( 12, label='ر', pos=( 1100, 495 ))
hurf.add_node( 13, label='ز', pos=( 1100, 540 ))
hurf.add_node( 14, label='س', pos=( 1100, 585 ))
hurf.add_node( 15, label='ش', pos=( 1100, 630 ))
hurf.add_node( 16, label='ص', pos=( 1100, 675 ))
hurf.add_node( 17, label='ض', pos=( 1100, 720 ))
hurf.add_node( 18, label='ط', pos=( 1100, 765 ))
hurf.add_node( 19, label='ظ', pos=( 1100, 810 ))
hurf.add_node( 20, label='ع', pos=( 1100, 855 ))
hurf.add_node( 21, label='غ', pos=( 1100, 900 ))
hurf.add_node( 22, label='ڠ', pos=( 1100, 945 ))
hurf.add_node( 23, label='ف', pos=( 1100, 990 ))
hurf.add_node( 24, label='ڤ', pos=( 1100, 1035 ))
hurf.add_node( 25, label='ق', pos=( 1100, 1080 ))
hurf.add_node( 26, label='ک', pos=( 1100, 1125 ))
hurf.add_node( 27, label='ݢ', pos=( 1100, 1170 ))
hurf.add_node( 28, label='ل', pos=( 1100, 1215 ))
hurf.add_node( 29, label='م', pos=( 1100, 1260 ))
hurf.add_node( 30, label='ن', pos=( 1100, 1305 ))
hurf.add_node( 31, label='و', pos=( 1100, 1350 ))
hurf.add_node( 32, label='ۏ', pos=( 1100, 1395 ))
hurf.add_node( 33, label='ه', pos=( 1100, 1440 ))
hurf.add_node( 34, label='ء', pos=( 1100, 1485 ))
hurf.add_node( 35, label='ي', pos=( 1100, 1530 ))
hurf.add_node( 36, label='ی', pos=( 1100, 1575 ))
hurf.add_node( 37, label='ڽ', pos=( 1100, 1620 ))


def draw_graph(graph, pos):
    # edges
    if pos==None:
        pos = nx.spring_layout(graph)  # positions for all nodes
         
    labels = nx.get_node_attributes(graph, 'label')
    colors = nx.get_edge_attributes(graph,'color').values()
    
    nx.draw(graph, 
            pos,
            # nodes' param
            with_labels=True, labels=labels,
            node_color='orange',
            node_size=40,
            font_size=8,
            # edges' param
            edge_color=colors, 
            )

draw_graph(hurf, None)
draw_graph(hurf, nx.get_node_attributes(hurf,'pos'))
draw_graph(hurf, nx.spiral_layout(hurf))


