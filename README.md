# TF_CogLTX
CogLTX tensorflow版本，bert超过512文本处理，长文本数据普遍存在，且文本中包含的信息非常分散，难以使用滑动窗口[2]截断。 [论文地址](https://keg.cs.tsinghua.edu.cn/jietang/publications/NIPS20-Ding-et-al-CogLTX.pdf)   
问答提取，分类下次更新

## 数据来源
点击 [这里](https://github.com/Maluuba/newsqa) 复制项目到本地生成combined-newsqa-data-v1.json文件  
将生成combined-newsqa-data-v1.json文件放在newsqa文件夹下  
也可以 点击 [这里](https://www.kaggle.com/fstcap/combinednewsqadatav1json/download) 直接下载

## 运行
#### 创建虚拟环境安装依赖库
``` 
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```
#### 将文本分割
```
python newsqa/process_newsqa.py
```
在data文件夹下生成newsqa_test_roberta-base.pkl，newsqa_test_roberta-base.pkl
#### 使用newsqa_test_roberta-base.pkl数据集进行训练，newsqa_test_roberta-base.pkl进行预测
```
python run_newsqa.py
```
## 流程梳理
#### 分块（block）
> 1、将text按照换行符分割成段，并每段进行分词得到tokenize列表；  
> 2、每个question作为一个样本，查找出该question对应answer的段，并找出answer的起始位置对应该段落tokenize的index，answer没有分布在同一段就抛弃该问题;  
> 3、把question, text分割成63个tokenize的block:  
>> question_blocks:   
>>> [CLS+tokenize<sub>63</sub>+SEP,tokenize<sub>63</sub>+SEP, tokenize<sub>63</sub>+SEP]   
> 
>> text_blocks:   
>>> [tokenize<sub>63</sub>+SEP,tokenize<sub>63</sub>+SEP, tokenize<sub>63</sub>+SEP]   
> 
> 4、将blocks转换成ids，赋值给Block类的ids属性，answer所在段的所有Block的relevance赋值3，answer所在的Block添加start和end属性并赋值上以该block作为起点answer起止tokenize的index； 
#### 打分模型训练（introspector）
> 1、生成输入样本（inputs）同一个把question生成两个样本：
>> 样本1：从bufs中随机block tokenize的长度不超过512;  
>> 样本2：从bufs中的block属性relevance大于等于1随机，和qbufs的tokenize长度与512之差大于64，再从relevance=0的block随机出样本。
> 
> 2、生成样本标签（labels）: bloack的属性relevance大于等于1的tokenize标为1，其他为0；  
> 3、训练：
>> introspector模型详见models.py的类Introspector；  
>> 损失函数交叉熵；  
>> 优化函数可变学习率Adam；  
> 
> 4、训练后将训练的样本进行预测得到每个blcok内tokenize的得分，block的得分等于内部tokenize的均值，记录在block属性estimation；  
#### 抽取式问答训练（reasoner）
> 1、生成输入样本（inputs）同一个把question生成两个样本：  
> 从bufs中筛选出bloack的属性relevance大于等于1的正block反之为负block  
>> 样本1：qbufs与正block的tokenize长度与512之差大于64则从负block挑选estimation降序排序的前几位，按照block在原本text的顺序排列；  
>> 样本2：qbufs与正block的tokenize长度与512之差大于64则从负block随机，按照block在原本text的顺序排列；  
> 
> 2、生成样本标签（labels）:  将answer所在block内tokenize的起始索引和结束索引转化成目前样本内tokenize的索引;  
> 3、训练：
>> reasoner模型详见models.py的类QAReasoner；  
>> 损失函数交叉熵；  
>> 优化函数可变学习率Adam；  
#### 预测（prediction）
> 1、text分割tokenize列表长度63的block，每个把question和text中所有block分别组成小于512的样本；  
> 2、所有样本通过训练后的introspector模型进行打分，挑选出前几位按照text原本顺序排序和question组成小于512的样本；  
> 3、该样本再次用introspector模型进行打分，question tokenize长度加上5的长度数量的样本数；  
> 4、该样本用reasoner模型预测得出answer的tokenize对应始末的索引；  
> 5、根据索引提取tokenize列表，转化成answer。
