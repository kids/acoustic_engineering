from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
secret_id = u'..'
secret_key = u'..'
region = u'ap-shanghai'
config = CosConfig(Secret_id=secret_id, Secret_key=secret_key, Region=region, Token=None)
client = CosS3Client(config)

import sys
import os
import asyncio
import threading
import numpy as np
from tqdm import tqdm
#import pandas as pd

def get_file_ext(sessid):
    try:
        fls=client.list_objects('..',Prefix=sessid)
    except:
        return ''
    if not 'Contents' in fls:
        return ''
    if len(fls['Contents'])==1:
        return 'json'
    fs=[j['Key'] for j in fls['Contents'] if not j['Key'].endswith('json')][0]
    return fs

async def pcm2wav(pcm_file, save_file, channels=1, bits=16, sample_rate=16000):
    import wave
    if type(pcm_file)==str:
        pcmf = open(pcm_file, 'rb')
        pcmdata = pcmf.read()
        pcmf.close()
    elif type(pcm_file)==bytes:
        pcmdata = pcm_file
    if bits % 8 != 0:
        raise ValueError("bits % 8 must == 0. now bits:" + str(bits))
    wavfile = wave.open(save_file, 'wb')
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits // 8)
    wavfile.setframerate(sample_rate)
    wavfile.writeframes(pcmdata)
    # return wavfile
    wavfile.close()

from aspeex import speex2wav

#a=list(set(open('../t_sess').read().split()))

def getall(a,ds):
    dn=[i[:-4] for i in os.listdir(f'/data9/wav/{ds}/')]
    for i in tqdm(a[:]):
        #if not i['data_type']=='getanswer_req':
        #    continue
        #if i['session_id'] in dn:
        if i in dn:
            continue
        #fs=get_file_ext(i['session_id'])
        fs=get_file_ext(i)
        if fs=='json':
            continue
        try:
            r=client.get_object('..',fs)
        except:
            continue
        loop = asyncio.get_event_loop()
        if fs.endswith('speex'):
            #speex2wav(r['Body'].get_raw_stream().data,'./wav/'+i+'.wav')
            task=asyncio.ensure_future(speex2wav(r['Body'].get_raw_stream().data,'/data9/wav/{}/{}.wav'.format(ds,i)))
            loop.run_until_complete(task)
            #t = threading.Thread(target=speex2wav,args=(fs,'./wav/'))
            #t.start()
        elif fs.endswith('pcm'):
            task=asyncio.ensure_future(pcm2wav(r['Body'].get_raw_stream().data,'/data9/wav/{}/{}.wav'.format(ds,i)))
            loop.run_until_complete(task)
        
