#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
train_list = open("/cache/icdar2013_word/gt.txt", 'r')
# print train_list.readlines()
train_lines = train_list.readlines()
print 'total-lines : ', len(train_lines)
dot_count = 0
blank_count = 0
counter = 0
max_len_fixed = 10
crop_cocotext_train = open("crop_icdar2013_val.lst", 'w')
char2idx = {}
char2idx[' '] = 0
idx2char = {}
idx2char[0] = ' '
vocab = []
vocab.append(' ')
idx = 1
max_len = 0
sig = 0
char2idx = json.load(open('char2idx.json', 'r'))
char2idx = json.JSONDecoder().decode(char2idx)
for line in train_lines[650:]:
    line = line[:-2]
    img_id = line.split(",")[0]
    text = line.split(",")[1][2:-1]
    print img_id, text
    if len(text) > max_len_fixed:
        continue
    max_len = max(max_len, len(text))
    label_string = ""
    text = text.decode('utf-8')
    for char in text:
        if char2idx.has_key(char):
            label_string += '\t' + str(char2idx[char])
        else:
            sig = 1
    if sig == 1 :
        sig = 0
        continue
    if len(text) < max_len_fixed:
        for i in range(max_len_fixed - len(text)):
            label_string += '\t' + str(0)
    line_to_lst = str(counter) + label_string + '\t' + img_id + '\n'
    counter += 1
    crop_cocotext_train.write(line_to_lst)
