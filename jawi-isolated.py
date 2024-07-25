letter = {
	0: ' ',
	1: 'ا',
	2: 'ب',
	3: 'ت',
	4: 'ة',
	5: 'ث',
	6: 'ج',
	7: 'چ',
	8: 'ح',
	9: 'خ',
	10: 'د',
	11: 'ذ',
	12: 'ر',
	13: 'ز',
	14: 'س',
	15: 'ش',
	16: 'ص',
	17: 'ض',
	18: 'ط',
	19: 'ظ',
	20: 'ع',
	21: 'غ',
	22: 'ڠ',
	23: 'ف',
	24: 'ڤ',
	25: 'ق',
	26: 'ک',
	27: 'ݢ',
	28: 'ل',
	29: 'م',
	30: 'ن',
	31: 'و',
	32: 'ۏ',
	33: 'ه',
	34: 'ء',
	35: 'ي',
	36: 'ی',
	37: 'ڽ'
}


def printnumber(thelist):
    kalima=''
    for num in thelist:
        kalima +=(letter[num])
    return(kalima)
    
import random

random_list = [random.randint(0, 38) for _ in range(20)]
    
print (printnumber(random_list))
print (printnumber([28,28,28,28,28]))

