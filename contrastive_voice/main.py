
import os,sys

import numpy as np

import torch

import librosa,scipy
from glob import glob

from network import LSTM
from loss import ConLoss


root_path='./ft_local/data/VoxCeleb1'

# make and save fbank for each reference
paths=glob(root_path+'/vox1_train_wav/*/*/')
for psec in paths[:2000]:
    fw=psec.replace('vox1_train_wav','vox1_train_fbank')
    if not os.path.exists(fw.rsplit('/',2)[0]):
        os.makedirs(fw.rsplit('/',2)[0])
    fs=os.listdir(psec)
    print(fw)
    mels=[]
    for f in fs:
        w=librosa.load(psec+f)
        mel=librosa.feature.melspectrogram(w[0],n_mels=32)
        mels.append(mel)
    np.save(fw[:-1]+'.npy',np.concatenate(mels,1))



# make dataset
paths=glob(root_path+'/vox1_train_fbank/*/')
batch_size=10
mel_len=64
sample_size=1000
dataset=[]
for _ in range(sample_size):
    for p in random.sample(paths,batch_size):
        secs=random.choices(os.listdir(p),k=2)
        batch_a=[]
        for sec in secs:
            sec=np.load(p+sec)
            start_t=random.sample(range(sec.shape[1]-mel_len),k=1)[0]
            batch_a.append(sec[:,start_t:start_t+mel_len])
        dataset.append(batch_a)
tdataset=torch.utils.data.TensorDataset(
    dataset,shuffle=True
)


# train
opt=torch.optim.SGD(lstm.parameters,lr=2e-1)

model=LSTM(32,128,128,2)
model.zero_grad()
for d in tdataset:
    embs,hidden=lstm(d[0],None)
embs.shape

c=ConLoss()
loss=c.forward(embs.view(1,2,128))

loss.backward()
opt.step()
model.zero_grad()



