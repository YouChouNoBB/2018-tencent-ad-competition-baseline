# coding=utf-8
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline
import pandas as pd

userFeature_data = []
with open('../data/userFeature.data', 'r') as f:
    cnt=0
    for i, line in enumerate(f):
        line = line.strip().split('|')
        userFeature_dict = {}
        for each in line:
            each_list = each.split(' ')
            userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
        userFeature_data.append(userFeature_dict)
        if i % 100000 == 0:
            print(i)
        if i % 1000000==0:
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv('../data/userFeature_'+str(cnt)+'.csv', index=False)
            cnt+=1
            del userFeature_data,user_feature
            userFeature_data=[]
    user_feature = pd.DataFrame(userFeature_data)
    user_feature.to_csv('../data/userFeature_' + str(cnt) + '.csv', index=False)
    del userFeature_data, user_feature
    user_feature=pd.concat([pd.read_csv('../data/userFeature_' + str(i) + '.csv') for i in range(cnt+1)]).reset_index(drop=True)
    user_feature.to_csv('../data/userFeature.csv', index=False)
