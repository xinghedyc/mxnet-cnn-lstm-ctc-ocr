#!/usr/bin/python
# -*- coding: UTF-8 -*-

train_list = open("/cache/icdar2013_word/gt.txt")
# print train_list.readlines()
train_lines = train_list.readlines()
print 'total-lines : ', len(train_lines)
dot_count = 0
blank_count = 0
counter = 0
max_len_fixed = 10
crop_cocotext_train = open("crop_icdar2013_train.lst", 'w')
char2idx = {}
char2idx[' '] = 0
idx2char = {}
idx2char[0] = ' '
vocab = []
vocab.append(' ')
idx = 1
max_len = 0
for line in train_lines[0:650]:
    line = line[:-2]
    img_id = line.split(",")[0]
    text = line.split(",")[1][2:-1]
    if len(text) > max_len_fixed:
        continue
    max_len = max(max_len, len(text))
    label_string = ""
    text = text.decode('utf-8')
    for char in text:
        if char not in vocab:
            vocab.append(char)
            print (char.encode('utf-8'))
            char2idx[char] = idx
            idx2char[idx] = char
            idx += 1
        label_string += '\t' + str(char2idx[char])
    if len(text) < 30:
        for i in range(max_len_fixed - len(text)):
            label_string += '\t' + str(0)
    line_to_lst = str(counter) + label_string + '\t' + img_id  + '\n'
    counter += 1
    crop_cocotext_train.write(line_to_lst)
print (char2idx)
import json
c2idx=json.JSONEncoder().encode(char2idx)
idx2c=json.JSONEncoder().encode(idx2char)
json.dump(c2idx,open('char2idx.json','w'))
json.dump(idx2c,open('idx2char.json','w'))
print "vacab : ", len(vocab)
print "train records : ", counter
