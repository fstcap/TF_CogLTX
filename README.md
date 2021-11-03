# TF_CogLTX
CogLTX tensorflow版本，bert长文本处理
问答提取部分，分类部分下次更新

## 数据来源
点击 [这里](https://github.com/Maluuba/newsqa) 复制项目本地生成combined-newsqa-data-v1.json文件  
将生成combined-newsqa-data-v1.json文件放在newsqa文件夹下

## 运行
首先运行newsqa/process_newsqa.py文件进行分块处理，data文件夹下会生成newsqa_test_roberta-base.pkl和newsqa_train_roberta-base.pkl
在运行run_newsqa.py进行训练预测
