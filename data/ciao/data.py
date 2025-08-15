
from scipy.io import loadmat
import random
import numpy as np
import math
import os
from collections import Counter
import sys
import pickle

random_seed = 0
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms

effective_scoring = 3
ratio = 0.8 
minimum_record = 3

print('raw data')
data = loadmat('rating.mat')
print(data.keys())
rating = data['rating']
print('Number fo UI record', rating.shape)

data = []
user_set = set()
for i in range (rating.shape[0]):
    if rating[i,3] > effective_scoring:
        uid = rating[i,0]
        iid = rating[i,1]
        data.append([uid, iid])
        if uid not in user_set:
            user_set.add(uid)

print(len(data))

user_items = {}
for d in data:
    user = d[0]
    item = d[1]
    if user not in user_items:
        user_items[user] = set()
        user_items[user].add(item)
    else:
        if item not in user_items[user]:
            user_items[user].add(item)


user_count = {}
item_count = {}
for i in range(len(data)):
    uid = data[i][0]
    iid = data[i][1]
    if uid not in user_count:
        user_count[uid] = 0
    if iid not in item_count:
        item_count[iid] = 0
    user_count[uid] += 1
    item_count[iid] += 1
    
print('Number of user in UI grph: ', len(user_count))
print('Number of item in UI grph: ', len(item_count))

print()
user_set = set()
item_set = set()
print('minimum record: ', minimum_record)
for k, v in user_count.items():
    if (v > minimum_record):
        user_set.add(k)

for k, v in item_count.items():
    if v > minimum_record:
        item_set.add(k)
        
print('Number of user',len(user_set))
print('Max user ID',max(user_set))
print('Number of item',len(item_set))
print('Max item ID',max(item_set))

user_reid_dict = dict(zip(list(user_set), list(range(len(user_set)))))
item_reid_dict = dict(zip(list(item_set), list(range(len(item_set)))))
user_set = set(user_reid_dict.values())
item_set = set(item_reid_dict.values())
print(len(user_set))  
print(max(user_set))  
print(len(item_set))  
print(max(item_set))  

print()
data_all = []
for i in range(len(data)):
    uid = data[i][0]
    iid = data[i][1]
    if uid in user_reid_dict and iid in item_reid_dict:
        data_all.append([user_reid_dict[uid], item_reid_dict[iid]])
print('all data:', len(data_all)) 

print('calculate item freq')
item_ids = [item[1] for item in data_all]
item_freq = Counter(item_ids)

# with open('item_freq_'+ str(minimum_record) +'.txt', 'w') as f:
#     for item_id, freq in item_freq.items():
#         f.write(f"{item_id} {freq}\n")
# print('saved')

print('train test split')
user_set = set(user_reid_dict.values())
item_set = set(item_reid_dict.values())

user_items_dict = {}
for i in range(len(data_all)):
    uid = data_all[i][0]
    iid = data_all[i][1]
    if uid in user_items_dict:
        user_items_dict[uid].append(iid)
    else:
        user_items_dict[uid] = [iid]
        
print('Number of user', len(user_items_dict))

train =[]
val = []
test = []
for user, items in user_items_dict.items():
    random.shuffle(items)
    if len(items[int(ratio * len(items)):]) >= 2:
        train_items = items[:int(ratio * len(items))]
        val_items = items[int(ratio * len(items)): int(0.9 * len(items))]
        test_items = items[int(ratio * len(items)):]
    else:
        train_items = items
        test_items = []  
    
    train.append([user] + train_items)
    test.append([user] + test_items)
    val.append([user] + val_items)
    
print("current work file:", os.getcwd())

with open('train_'+str(minimum_record)+'.txt', 'w') as file:
    for sublist in train:
        line = ' '.join(map(str, sublist))  
        file.write(line + '\n')  
print("Train data saved successfully:")

with open('test_'+str(minimum_record)+'.txt', 'w') as file:
    for sublist in test:
        line = ' '.join(map(str, sublist))  
        file.write(line + '\n') 
print("Test data saved successfully:")
        
with open('val_'+str(minimum_record)+'.txt', 'w') as file:
    for sublist in val:
        line = ' '.join(map(str, sublist))  
        file.write(line + '\n') 
print("Val data saved successfully:")
'''

lengths = [len(sublist) for sublist in train]
max_len = max(lengths)
min_len = min(lengths)
mean_len = np.mean(lengths)
median_len = np.median(lengths)


print(f"{max_len}")        
print(f"{min_len}")        
print(f"{mean_len:.2f}")   
print(f"{median_len}")  

print()
print('Number of user',len(user_set))
print('Max user ID',max(user_set))     
print('Number of item',len(item_set))
print('Max item ID',max(item_set))

data_info = {'num_users': ,   # fill according to the data info
             'num_items': ,
             'pad_item': }

with open('ciao_info.pkl', 'wb') as f:
    pickle.dump(data_info, f)
print('saved')
'''