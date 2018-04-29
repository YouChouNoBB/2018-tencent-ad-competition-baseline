## 项目介绍

- 2018腾讯广告大赛baseline 100行代码带你上0.73
- 比赛报名地址：algo.qq.com/person/mobile/landingPage?from=dsbryan
![Image text](https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline/blob/master/pic/leadboard.jpg)

- baseline  在内存大于32g的情况下使用，挑整参数，可以上0.74 
- 线上0.74可以参考：https://pan.baidu.com/s/1o8ntpTnZITjZtwfhfIIOkg
- 1.首先处理4个G的用户特征

    因为数据太大，而且不是能直接pandas读取的格式，所以需要做格式转换，用dict的方式来初始化DataFrame

- 2.拼接用户特征，广告特征

      训练数据中负样本的标签给的是-1，需要先转成0，预测数据的标签置为-1，方便合并后区分数据集。将缺失值填充为 '-1' ，为什么不是数值的-1呢？因为在LabelEncoder的时候需要对数据排序，同时存在string和int类型是无法比较的。所以需要填充为string类型的 ‘-1’。

- 3.将单取值的离散特征使用稀疏方式one-hot

      为什么要先将数据划分为训练集和测试集呢，因为稀疏的数据是无法分片的，所以只能先划分数据，分别拼接稀疏特征。如果使用pd.get_dummy()来获取onehot特征，生成的数据是可以用来分片的，但是稠密存储是个致命弱点。
      github上很多人问我train_x=train[['creativeSize']] 这句是什么意思，其实creativeSize这个特征是数值特征，不需要进行特别的处理，如果想处理的话可以考虑pd.cut来分段离散化。另一个原因是把这个特征拿出来构造一个新的DataFrame，方便和后面生成的稀疏特征进行拼接。所以使用的是[[]]来取值获得一个DataFrame，而不是[]取值来或者一个Seris

- 4.将多取值的离散特征使用稀疏方式向量化

      这个操作估计很多同学之前没有见过，一般出现在自然语言处理中，计算TF-IDF，LDA等时候使用，但是同样可以用来生成一个稀疏向量，作为新的特征，同时可以一个特征生成多个特征，比单独的处理更加方便。

- 5.线下测试

      使用train_test_split划分数据，这行注释掉了。

- 6.线上提交

      线上预测的时候，模型训练中early_stopping_rounds 这个参数没什么用，参数n_estimatorsxu需要根据线下测试来重新指定。我看到有些同学设置为10000取得了0.74的成绩。。。。

##
- baseline_v2  16g左右内存使用，建议开启虚拟内存
- 将训练集分片，分别训练预测，然后将预测结果求平均，大概需要15g内存

##
- baseline_v3  只要能读进去数据，join完应该就能跑了
- 每次训练重新读取的数据，这样应该8g内存就能跑了

##
--可以先使用 user_feature_tocsv.py 将用户特征转换成csv文件，以便后面直接pd.read_csv读入
