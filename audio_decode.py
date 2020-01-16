#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, time, json, base64
import numpy as np
import opuslib

channels = 1
rate = 16000
frame_size = 960


def extract_opusbytes(x):
    # input: sorted base64 encoded slices
    ox=[base64.b64decode(i) for i in x]
    tdata=[]
    for i in range(len(ox)):
        t=ox[i]
        tindex=0
        while tindex<len(t):
            tlen=np.fromstring(t[tindex:tindex+4][::-1],np.int32)[0]
            print(tlen,end=',')
            tdata.append(t[tindex+8:tindex+8+tlen])
            tindex=tindex+8+tlen
    return tdata

def decode_opus(x):
    # input: pure opus slices
    decoder = opuslib.Decoder(rate, channels)
    pcms=[decoder.decode(i,frame_size) for i in x]
    return b''.join(pcms)

def decode_fmt(x):
    # input: json.loads(voicedata_log_fmt.select('vec_data_base64_str').take(1)[0]['vec_data_base64_str'])
    decoder = opuslib.Decoder(rate, channels)
    pcms=[decoder.decode(base64.b64decode(i),frame_size) for i in x]
    return b''.join(pcms)
    

if __name__ == '__main__':
    x=[ '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',
      ]
    opusx = extract_opusbytes(x)
    pcm = decode_opus(opusx)
    print(pcm[:10])
    
    y=[
     'SC1mdwPPqxNd+20rRCn/sKhGD54TqOEfZi4cxpQlJqPu/MkPjjH0',
     'SIjXyQg9PYdICLEUpSu+ATsOJM8reUzVs16l9DbDoOfgeND9BTUOo5NkGn4=',
     'SIyzkLVMxvQ2ThxqOfH2OQyrmRE0xXQ5Su5/J/0A8fLRNj+RyiFYdLzz+1i2iwylQUvmqsjQ',
     'SI/0LSKLuuBxuqipLx6EQREu8Wgx+EpetqRYnR6oeoBhQeJEM6jAd+uFOmyqeYgK8g==',
     'SJF4C2ggUETdtwUs9vQPeM6fAwDyuWvyTheixLRnQIHfkMb+5C/yCLkUhiRVkvVGPD1rsxMN4y6o/qk=',
     'SJFE/4n+cAOiklFbd1dzs8tZtTz+kpb4DyiG7A+SgV1m0jzmsKFEctAfbbKXOV/rlBP/',
     'SI8G2HqocgqLcaO4ItFZc8gCSYDg1KmDc2NvmkIQ+4MHsURDRbbtYkfDuVAwUqvHoAOCWUNg',
     'SKSp0Aq7OJuw7txxqpvsX64On4WAaqHo4a77xzpJkS4P3MXVoWLSaD040hN+LJ6Vqo3g',
     'SKDDB+BNJnxYqHiYAGhGChFw2LH0WaZTFfkxfkVzeW+CzR8nF6CmD6w/AAu5kiwzWw4=',
     'SJ0Ha7RBgOV8iA0Fs3/iqrniMONtQB1et8qY8OQ3bV52qFdKWyt9RfLVpvoOYUYHLkA=',
     'SJwfdDrlczeQZ6TO6utxR8mw/t7jYPXaf1RP7DrJrHAcFClD/tJI2o5YjDUXRuA=',
     'SJujCDV65i6RLnBFem1CcVNMXGmujXE6wSrqn4TjWDwhs1rkOersGV94',
     'SImdxMWrf7oXflTuhKBa64ueCRAal3NYQKA/zwxk7nZpYNbNib1onsNMkw==',
     'SIHTxJpUxWrrtLB8F3y1/VG0s7IU/va8B0N+h0ZGcWJM4fbJjQdjYqGbmPKR4MUVY3KTym5a0hHWrIY=',
     'SKz+Ujg6gn9EXDfhcxHRN/YVrSOoKrM/F1Q1G+ekeY+rwohUVVIOl3O1CbWE9wPsM4Q4Vig9bBPXx5FXGoA=',
     'SKxtmtx0BOyV8ZHwCdmX9Mh4NGcibaY1WCuynp49692TQjw6ocnayn/FTFS2m7kiOIA=',
     'SKliJoLaktYZNwfwY3YPJlDU83QF7OVaXyrno6fkXkYJNNGxUYgqpX5KxDCRslklSyynwA==',
     'SKX9Lxqd/4tw8CnlV2XxWjR8vr7geNhtnCAQ5loEz4d/SFG4a2lWkleEXMueL7hHgA==',
     'SKUkhxrT1oQuxjTu0F0zRB+PVJ1eZoEtYXhDLajx/AQYDT+7ExEQ+/tdugeVWHY/GQ==',
     'SKUyokNKNQk8oayd7LpAi+7fsiV1w2nknt4B0wFP28XEdab8HdEg66ABmFL9+nhxVCY1xRtwqc3A',
     'SKaaWFl1by7X6Npz+lSHqgtSyhcLbi3NtUXn9y58sDakEAlnPunYyYIP1A/KCUMn3oA7aitq3i0g',
     'SK01cibSUNFvAUk6yy/tpX2PSF57OtvE7gZF7rHle5dMLpsDWmXdD2btphNdj8H1TFpE5Wa8k17DqIWVyIw=',
     'SK6f8S0wQV64Cotnn5ebo3pRs/fvYI0y7uzA7lgT6TzMsAovM+4S+/m5Poi+aTCl68zAwGqokwv9gA==',
     'SK33I9G9SCdKILk39HXyfn3atW/8bNht64GsjDa7X3K49SgqVHP0ir9PX8DysMNoWR5SKBlBSARo',
     'SIzRljZSng/4JjjMmWP5uMARrh0qbe9VPFS4nveVL86A']
     pcm = decode_fmt(y)
     print(y[:10])


