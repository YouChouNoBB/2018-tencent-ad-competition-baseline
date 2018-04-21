## 项目介绍

- 2018腾讯广告大赛baseline 100行代码带你上0.73
![Image text](https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline/blob/master/pic/leadboard.jpg)

- baseline  在内存大于32g的情况下使用，挑整参数，可以上0.74
- 1.首先处理4个G的用户特征
- 2.拼接用户特征，广告特征
- 3.将单取值的离散特征使用稀疏方式one-hot
- 4.将多取值的离散特征使用稀疏方式向量化
- 5.线下测试
- 6.线上提交

##
- baseline_v2  16g左右内存使用，建议开启虚拟内存
- 将训练集分片，分别训练预测，然后将预测结果求平均，大概需要15g内存
- 如果还觉得内存占用过多，可以尝试每次join完之后del 掉用户特征，下次训练再重新读取，这样应该8g内存就能跑了