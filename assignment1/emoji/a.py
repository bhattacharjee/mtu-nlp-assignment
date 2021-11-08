#!/usr/bin/python3

import re

with open("temp2.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]", flags=re.UNICODE)

    for line in lines:
        print(line, "->", re.sub(emoji_pattern, 'EMOJI', line))

    for line in lines:
        print(line, "->", emoji_pattern.sub(line, 'EMOJI'))

    print("Other test", re.compile('This').sub(line, 'NOT THIS'))

