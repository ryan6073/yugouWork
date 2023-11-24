1.0:（朱家顺）提交初始版本，提供了完备的特征工程建立，和MLPClassifier分类器，效果图：![效果图](../../桌面/yugouWork/img/initial.png)
1.1:（张家豪）提交第二个版本，效果图：![效果图](../../桌面/yugouWork/img/zjh1.jpg)
1.2:（毛南）提交第三个版本，

对user_log中brand缺失数据进行填充处理

调整训练集和数据集的划分，使80%用于训练

年龄范围和性别缺失值使用均值填充

在决策树上使用adaboost分类器

使用网格搜索提供最优参数

效果图：

![效果图](../../桌面/yugouWork/img/mn1.jpg)

1.3：(苏沛泽)提交第四个版本，

考虑将多个模型融合，利用Stacking堆叠法，将两层算法进行串联。

在第一层中首先将多层感知机模型、逻辑斯特模型、决策树模型、随机森林模型、梯度提升回归树模型作为基础模型进行训练。

训练完后进行预测，得到预测结果后，将其预测结果转成二维矩阵作为一组新的特征交由第二层元学习器进行训练。

元学习器这里采用毛南设计好的adaboost分类器，通过将第一层基础模型的预测结果作为新的特征矩阵进行训练。

融合模型最终输出的预测结果就是元学习器输出的结果。

效果图：

![image-20231124183325080](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231124183325080.png)

1.4（翟启发）提交第五个版本

​	这段代码通过梯度提升树（Gradient Boosting Tree）进行分类任务，首先训练了模型并进行特征选择，然后利用网格搜索（Grid Search）调整模型参数以提高预测准确率，并最终评估了模型在测试集上的性能。

![zhai1](D:\Gitcode\yugouWork\img\zhai1.png)

![zhai2](D:\Gitcode\yugouWork\img\zhai2.png)

![zhai3](D:\Gitcode\yugouWork\img\zhai3.png)

![zhai4](D:\Gitcode\yugouWork\img\zhai4.png)
