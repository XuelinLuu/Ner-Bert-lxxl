#### Ner-Bert-lxxl

---
- 本项目是将BERT模型应用于命名实体识别
- 将标注任务分类为['B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE', 'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE', 'O', 'S-NAME', 'S-ORG', 'S-RACE']19个类别，分别进行训练标注

---

#### 文件介绍
- bert-base-chinese-local 是bert预训练模型数据，项目中已经全部改为bert-base-chinese，如果无法通过项目内下载可以改为本路径
- datasets 保存本项目中的所有训练数据，验证数据以及测试数据
- models 为本项目fine-tune之后的模型保存
- src
  - config.py 参数定义文件
  - datasets.py 数据预处理文件
  - engine.py 损失、训练、评估
  - model.py 项目的模型
  - run_train.py 训练
  - run_eval.py 评估

#### 项目运行

- 训练直接运行run_train.py
- 评估直接运行run_eval.py