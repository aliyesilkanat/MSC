#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

def read_speaker_info(speaker_info_path):
    speaker_ids = []
    with open(speaker_info_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            speaker_id = line.strip().split()[0]
            speaker_ids.append("p"+speaker_id)
    return speaker_ids


def read_filenames(root_dir):
    speaker2filenames = defaultdict(lambda : [])
    for path in sorted(glob.glob(os.path.join(root_dir, '*/*'))):
        filename = path.strip().split('/')[-1]
        speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
        speaker2filenames["p"+speaker_id].append(path)
    return speaker2filenames

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


# In[3]:


vctk_ids=read_speaker_info("/raid/users/ayesilkanat/MSC/VCTK/VCTK-Corpus/speaker-info.txt")


# In[5]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(vctk_ids, test_size=0.2, random_state=13)

train, val = train_test_split(train, test_size=0.2, random_state=1)


# In[6]:


train_speaker_ids=train + [os.path.split(path)[-1] for path in sorted(glob.glob("/raid/users/ayesilkanat/MSC/SELL-CORPUS/train/*/*"))]



test_speaker_ids=[os.path.split(path)[-1] for path in sorted(glob.glob("/raid/users/ayesilkanat/MSC/SELL-CORPUS/dev/*/*"))]


# In[14]:


stage=0
segment_size=128
n_out_speakers=20
test_prop=0.1
sample_rate=24000
training_samples=10000000
testing_samples=10000
n_utt_attr=5000


output_dir = "../spectrograms/sr_24000_mel_norm_128frame_256mel"
test_proportion = test_prop

n_utts_attr = n_utt_attr

#$raw_data_dir/wav48 $raw_data_dir/speaker-info.txt $data_dir $n_out_speakers $test_prop $sample_rate $n_utt_attr


speaker2filenames = defaultdict(lambda : [])
for path in sorted(glob.glob(os.path.join("/raid/users/ayesilkanat/MSC/VCTK/VCTK-Corpus/wav48", '*/*'))):
    filename = path.strip().split('/')[-1]
    speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
    if speaker_id in train:
        speaker2filenames["p"+speaker_id].append(path)


for folder_path in sorted(glob.glob("/raid/users/ayesilkanat/MSC/SELL-CORPUS/train/*/*")):
    speaker_id=os.path.split(folder_path)[-1]
    paths=glob.glob(os.path.join(folder_path,"*.wav"))
    for path in paths:
        speaker2filenames[speaker_id].append(path)

for folder_path in sorted(glob.glob("/raid/users/ayesilkanat/MSC/SELL-CORPUS/dev/*/*")):
    speaker_id=os.path.split(folder_path)[-1]
    paths=glob.glob(os.path.join(folder_path,"*.wav"))
    for path in paths:
        speaker2filenames[speaker_id].append(path)

        
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


# In[15]:


speaker2filenames


# In[ ]:





# In[8]:




with open(os.path.join(output_dir, 'in_test_files.txt'), 'w') as f:
    for path in in_test_path_list:
        f.write(f'{path}\n')



with open(os.path.join(output_dir, 'out_test_files.txt'), 'w') as f:
    for path in out_test_path_list:
        f.write(f'{path}\n')


# In[12]:


train_path_list


# In[9]:




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


# In[ ]:


del data
gc.collect()


# In[ ]:





# In[ ]:



pkl_path = "../spectrograms/sr_24000_mel_norm_128frame_256mel/train.pkl"
output_path = "../spectrograms/sr_24000_mel_norm_128frame_256mel/train_"+str(segment_size)+".pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

reduced_data = {key:val for key, val in data.items() if val.shape[0] > segment_size}

with open(output_path, 'wb') as f:
    pickle.dump(reduced_data, f)


# In[ ]:


del reduced_data
gc.collect()


# In[ ]:





pickle_path = "../spectrograms/sr_24000_mel_norm_128frame_256mel/train.pkl"
sample_path = "../spectrograms/sr_24000_mel_norm_128frame_256mel/train_samples_"+str(segment_size)+".json"
n_samples = training_samples

sample_single_segments(pickle_path,sample_path,segment_size,n_samples)
gc.collect()


# In[ ]:





pickle_path = "../spectrograms/sr_24000_mel_norm_128frame_256mel/in_test.pkl"
sample_path = "../spectrograms/sr_24000_mel_norm_128frame_256mel/in_test_samples_"+str(segment_size)+".json"
n_samples = testing_samples

sample_single_segments(pickle_path,sample_path,segment_size,n_samples)
gc.collect()


# In[ ]:





pickle_path = "../spectrograms/sr_24000_mel_norm_128frame_256mel/out_test.pkl"
sample_path = "../spectrograms/sr_24000_mel_norm_128frame_256mel/out_test_samples_"+str(segment_size)+".json"
n_samples = testing_samples
sample_single_segments(pickle_path,sample_path,segment_size,n_samples)
gc.collect()


# In[ ]:





# In[ ]:




