#!/usr/bin/python
# -*- coding: UTF-8 -*-

LINES_LED = [
	[0, 77], 	    #0
	[311, 234], 	#1
	[312, 389], 	#2
	[546, 623], 	#3
	[545, 468], 	#4
	[78, 155], 	    #5
	[156, 233], 	#6
	[390, 467], 	#7
	[624, 701], 	#8
	[702, 779], 	#9
	[1559, 1482], 	#10
	[780, 857],  	#11
	[935, 858],  	#12
	[936, 1013],  	#13
	[1091, 1014],  	#14
	[1092, 1169],  	#15
	[1247, 1170],  	#16
	[1248, 1325],  	#17
	[1403, 1326],  	#18
	[1404, 1481],  	#19
	[1950, 2027],  	#20
	[2184, 2261],  	#21
	[2262, 2339],  	#22
	[1715, 1638],  	#23
	[1637, 1560],  	#24
	[1949, 1872],  	#25
	[2183, 2106],  	#26
	[2028, 2105],  	#27
	[1716, 1793],  	#28
	[1794, 1871]  	#29
]

LINES_LED_INDEX = []

min_v = 0
max_v = 77
for x in range(0, len(LINES_LED)):
    for i in range(0, len(LINES_LED)):
        item = LINES_LED[i]
        if (item[0] == min_v or item[1] == min_v):
            LINES_LED_INDEX.append(i)
            min_v += 78
            max_v += 78
            continue
print(LINES_LED_INDEX)

LINES_LED_NEW = list(range(0, len(LINES_LED)))
min_v = 0
max_v = 75
for x in LINES_LED_INDEX:
    item_new = [0, 0]
    item = LINES_LED[x]
    if item[0] > item[1]:
        item_new[0] = max_v
        item_new[1] = min_v
    else:
        item_new[0] = min_v
        item_new[1] = max_v
    LINES_LED_NEW[x] = item_new
    min_v += 76
    max_v += 76

for x in range(0, len(LINES_LED_NEW)):
    item = LINES_LED_NEW[x]
    print(item, ',       #', x)
