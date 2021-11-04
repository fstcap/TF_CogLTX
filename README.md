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