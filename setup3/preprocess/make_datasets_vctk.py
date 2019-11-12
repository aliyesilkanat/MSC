#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pickle 
import librosa 
import sys
import glob 
import random
import os
from collections import defaultdict
import re
import numpy as np
import json
from tacotron.utils import get_spectrograms
import pandas as pd
import gc 
from concurrent.futures import ThreadPoolExecutor

def read_speaker_info(speaker_info_path):
    speaker_ids = []
    with open(speaker_info_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            speaker_id = line.strip().split()[0]
            speaker_ids.append(speaker_id)
    return speaker_ids


def wave_feature_extraction(wav_file, sr):
    y, sr = librosa.load(wav_file, sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y

def spec_feature_extraction(wav_file):
    mel, mag = get_spectrograms(wav_file)
    return mel, mag 


def sample_single_segments(pickle_path,sample_path,segment_size,n_samples):

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # (utt_id, timestep, neg_utt_id, neg_timestep)
    samples = []

    # filter length > segment_size
    utt_list = [key for key in data]
    utt_list = sorted(list(filter(lambda u : len(data[u]) > segment_size, utt_list)))
    print(f'{len(utt_list)} utterances')
    sample_utt_index_list = random.choices(range(len(utt_list)), k=n_samples)

    for i, utt_ind in enumerate(sample_utt_index_list):
        if i % 500 == 0:
            print(f'sample {i} samples')
        utt_id = utt_list[utt_ind]
        t = random.randint(0, len(data[utt_id]) - segment_size)
        samples.append((utt_id, t))

    with open(sample_path, 'w') as f:
        json.dump(samples, f)


# In[27]:


l1arctic_training_ids=[a[44:47] for a in sorted(glob.glob("/raid/users/ayesilkanat/MSC/L1Arctic/*_arctic"))]
l2arctic_training_ids=['ABA','SKA','BWC','LXC','ASI','SVBI','HKK','HJK','EBVS','MBMPS','HQTV','PNV']
l1arctic_test_speaker_ids=[l1arctic_training_ids.pop(),l1arctic_training_ids.pop()]
l1arctic_test_speaker_ids.pop() # leave one for validation one for test
l2arctic_test_speaker_ids=['YBAA', 'NCC', 'RRBI', 'YDCK', 'ERMS', 'THV']
test_speaker_ids = l1arctic_test_speaker_ids +l2arctic_test_speaker_ids 
train_speaker_ids = l1arctic_training_ids + l2arctic_training_ids

print("Training ids: ",train_speaker_ids)
print("Testing ids: ",test_speaker_ids)


# In[28]:



stage=0
segment_size=128
n_out_speakers=20
test_prop=0.1
sample_rate=24000
training_samples=10000000
testing_samples=10000
n_utt_attr=5000

output_dir = "/raid/users/ayesilkanat/MSC/adaptive-accent-conversion/setup3/spectrograms/sr_24000_mel_norm_128frame_256mel"
test_proportion = test_prop

n_utts_attr = n_utt_attr


speaker2filenames = defaultdict(lambda : [])


for tr_id in l1arctic_training_ids:
    for path in sorted(glob.glob("/raid/users/ayesilkanat/MSC/L1Arctic/cmu_us_"+tr_id+"_arctic/wav/*.wav")):
        speaker2filenames[tr_id].append(path)


for tst_id in l2arctic_training_ids:
        
    for path in sorted(glob.glob("/raid/users/ayesilkanat/MSC/ArcticL2/"+tst_id+"/wav/*.wav")):
        speaker2filenames[tst_id].append(path)
        
for tr_id in l1arctic_test_speaker_ids:
    for path in sorted(glob.glob("/raid/users/ayesilkanat/MSC/L1Arctic/cmu_us_"+tr_id+"_arctic/wav/*.wav")):
        speaker2filenames[tr_id].append(path)
      
        
for tst_id in l2arctic_test_speaker_ids:
        
    for path in sorted(glob.glob("/raid/users/ayesilkanat/MSC/ArcticL2/"+tst_id+"/wav/*.wav")):
        speaker2filenames[tst_id].append(path)
        
        
        
train_path_list, in_test_path_list, out_test_path_list = [], [], []
for speaker in train_speaker_ids:
    path_list = speaker2filenames[speaker]
    random.shuffle(path_list)
    test_data_size = int(len(path_list) * test_proportion)
    train_path_list += path_list[:-test_data_size]
    in_test_path_list += path_list[-test_data_size:]
for speaker in test_speaker_ids:
    path_list = speaker2filenames[speaker]
    out_test_path_list += path_list
    
    
print(speaker2filenames.keys())


# In[ ]:





# In[9]:




with open(os.path.join(output_dir, 'in_test_files.txt'), 'w') as f:
    for path in in_test_path_list:
        f.write(f'{path}\n')



with open(os.path.join(output_dir, 'out_test_files.txt'), 'w') as f:
    for path in out_test_path_list:
        f.write(f'{path}\n')


# In[23]:



#with ThreadPoolExecutor(max_workers=256) as executor:
#    future = executor.submit(spec_feature_extraction, train_path_list[0])
#    print(future.result()[1])
    


# In[ ]:




for dset, path_list in zip(['train', 'in_test', 'out_test'],         [train_path_list, in_test_path_list, out_test_path_list]):
    print(f'processing {dset} set, {len(path_list)} files')
    data = {}
    output_path = os.path.join(output_dir, f'{dset}.pkl')
    all_train_data = []
    for i, path in enumerate(sorted(path_list)):
        if i % 500 == 0 or i == len(path_list) - 1:
            print(f'processing {i} files')
        filename = path.strip().split('/')[-1]
        mel, mag = spec_feature_extraction(path)
        data[filename] = mel
        if dset == 'train' and i < n_utts_attr:
            all_train_data.append(mel)
    if dset == 'train':
        all_train_data = np.concatenate(all_train_data)
        mean = np.mean(all_train_data, axis=0)
        std = np.std(all_train_data, axis=0)
        attr = {'mean': mean, 'std': std}
        with open(os.path.join(output_dir, 'attr.pkl'), 'wb') as f:
            pickle.dump(attr, f)
    for key, val in data.items():
        val = (val - mean) / std
        data[key] = val
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


# In[99]:


del data
gc.collect()


# In[100]:





# In[ ]:



pkl_path = os.path.join(output_dir,"train.pkl")
output_path = os.path.join(output_dir,"train_"+str(segment_size)+".pkl")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

reduced_data = {key:val for key, val in data.items() if val.shape[0] > segment_size}

with open(output_path, 'wb') as f:
    pickle.dump(reduced_data, f)


# In[ ]:


del reduced_data
gc.collect()


# In[ ]:





pickle_path = os.path.join(output_dir,"train.pkl")
sample_path = os.path.join(output_dir,"train_samples_"+str(segment_size)+".json")
n_samples = training_samples

sample_single_segments(pickle_path,sample_path,segment_size,n_samples)
gc.collect()


# In[ ]:





pickle_path = os.path.join(output_dir,"in_test.pkl")
sample_path = os.path.join(output_dir,"in_test_samples_"+str(segment_size)+".json")
n_samples = testing_samples

sample_single_segments(pickle_path,sample_path,segment_size,n_samples)
gc.collect()


# In[ ]:





pickle_path = os.path.join(output_dir,"out_test.pkl")
sample_path = os.path.join(output_dir,"out_test_samples_"+str(segment_size)+".json")
n_samples = testing_samples
sample_single_segments(pickle_path,sample_path,segment_size,n_samples)
gc.collect()


# In[ ]:


2+2


# In[ ]:




