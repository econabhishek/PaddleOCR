###Check unique number of characters in names 

import pandas as pd


df = pd.read_csv('C:/Users/ASUS/Dropbox/Melissa_Dell/Paddle_test_images/multi_lang_gt/JP_ja_alt_names.csv')

##Load charset to check
charset_paddle=('C:/Users/ASUS/Dropbox/Melissa_Dell/Codebase/PaddleOCR/ppocr/utils/dict/japan_dict.txt')

##Load charset sep by \n to list
with open(charset_paddle, 'r') as f:
    charset_paddle = f.read().splitlines()


###Convert first column to list 
names = df['name'].tolist()

##Get all unique chracters used in all elements of names
unique_chars = set(''.join(names))

print(len(unique_chars), "Unique characters used in names")
print(unique_chars)


##Check if all unique characters are in charset
count=0
for char in unique_chars:
    if char not in charset_paddle:
        print(char, "is not in charset")
        count+=1

print(count, "characters not in charset")
