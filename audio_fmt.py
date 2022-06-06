#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession, HiveContext
from pyspark.sql.types import *
from pyspark.sql.functions import lit
import sys, os, time, json, base64
import numpy as np


def f1(l):
    k,v=l
    v=sorted(v,key=lambda vk:vk['voice_offset_int'])
    x=[i['vec_data_base64_str'] for i in v]
    try:
        ox=[base64.b64decode(i) for i in x]
        tdata=[]
        for i in range(len(ox)):
            t=ox[i]
            tindex=0
            while tindex<len(t):
                tlen=np.fromstring(t[tindex:tindex+4][::-1],np.int32)[0]
                #print(tlen,end=',')
                tdata.append(t[tindex+8:tindex+8+tlen])
                tindex=tindex+8+tlen
        rx=json.dumps([base64.b64encode(i).decode() for i in tdata])
    except:
        print('EXCEPTION:',l)
        rx=''
    ret=[i for i in v[-1]]
    assert ret[6]==x[-1]
    ret[6]=rx
    return ret[:-1]


if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("roi") \
        .getOrCreate()
    ds=int(sys.argv[1])
    
    sc = spark.sparkContext
    hiveContext = HiveContext(sc)
    hiveContext.setConf("hive.exec.dynamic.partition", "true") 
    hiveContext.setConf("hive.exec.dynamic.partition.mode", "nonstrict")

    df = spark.sql('SELECT * FROM wecar.t_si_wecar_voicedata_log where ds>={0}00 and ds<{0}99 '.format(ds))
    df.show(2)
    ndf=df.rdd.filter(lambda k:not not k['session_id_str']) \
            .groupBy(lambda k:k['session_id_str']) \
            .map(f1).toDF(df.columns[:-1])
    ndf=ndf.withColumn('ds', lit(ds))
    ndf.show(2)
    ndf.write.save('hdfs://../ds={}'.format(ds),
             format='parquet', mode='overwrite')

