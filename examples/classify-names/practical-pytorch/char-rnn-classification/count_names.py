import torch
from data import *
from model import *
import random
import time
import math
import numpy as np

#build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('../data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

name_len_count = np.zeros(30)

for key, value in category_lines.items():
    for name in value:
        name_len_count[len(name)-1] = name_len_count[len(name)-1] + 1

name_len_count = name_len_count.astype(np.int)

print ('all_categories', all_categories)
#print ('category_lines', category_lines)
print ('name_len_count', name_len_count)
print ('name_len_count sum', np.sum(name_len_count))
