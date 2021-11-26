#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import typing

test_file = 'germeval2018.test.txt'
training_file = 'germeval2018.training.txt'

catlist = ['INSULT', 'OFFENSE', 'OTHER', 'ABUSE', 'PROFANITY']

def get_line_and_type(line:str):
    pat = re.compile('[A-Z]{2,}')

    words = line.split()
    categories = set()
    while len(words) > 0 and words[-1] in catlist:
        categories.add(words[-1])
        words = words[:-1]
    return " ".join(words), list(categories)

def is_toxic(categories:list)->int:
    for cat in ['INSULT', 'OFFENSE', 'ABUSE']:
        for w in categories:
            if w == cat:
                return 1
    return 0

def process_file(filename:str):
    s = set()
    text_arr = []
    toxic_arr = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line, cats = get_line_and_type(line)
            cats = is_toxic(cats)
            text_arr.append(line)
            toxic_arr.append(cats)
    return text_arr, toxic_arr


text1, cat1 = process_file(test_file)
text2, cat2 = process_file(training_file)
text = text1 + text2
cat = cat1 + cat2

df = pd.DataFrame({'comment_text': text, 'Sub1_Toxic': cat})
print(df)
df.to_csv('germeval2018_a.txt', index=False)
