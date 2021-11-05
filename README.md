# TF_CogLTX
CogLTX tensorflow版本，bert超过512文本处理
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
> 3、把question, text分割成63个tokenize的block。  
> question_blocks:  
> [CLS+tokenize<sub>63</sub>+SEP,tokenize<sub>63</sub>+SEP, tokenize<sub>63</sub>+SEP]  
> text_blocks:  
> [tokenize<sub>63</sub>+SEP,tokenize<sub>63</sub>+SEP, tokenize<sub>63</sub>+SEP]  
> 4、将blocks转换成ids，赋值给Block类的ids属性，answer所在段的所有Block的relevance赋值3，answer所在的Block添加start和end属性并赋值上以该block作为起点answer起止tokenize的index；  