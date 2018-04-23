## 项目介绍

- 2018腾讯广告大赛baseline 100行代码带你上0.73
- 比赛报名地址：algo.qq.com/person/mobile/landingPage?from=dsbryan
![Image text](https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline/blob/master/pic/leadboard.jpg)

- baseline  在内存大于32g的情况下使用，挑整参数，可以上0.74 
- 线上0.74可以参考：https://pan.baidu.com/s/1o8ntpTnZITjZtwfhfIIOkg
- 1.首先处理4个G的用户特征
- 2.拼接用户特征，广告特征
- 3.将单取值的离散特征使用稀疏方式one-hot
- 4.将多取值的离散特征使用稀疏方式向量化
- 5.线下测试
- 6.线上提交

##
- baseline_v2  16g左右内存使用，建议开启虚拟内存
- 将训练集分片，分别训练预测，然后将预测结果求平均，大概需要15g内存

##
- baseline_v3  只要能读进去数据，join完应该就能跑了
- 每次训练重新读取的数据，这样应该8g内存就能跑了
