# 项目结构

```bash
project_root/
├── run.py             # 入口脚本：训练 / 评估
├── train.py           # 训练循环
├── evaluate.py        # BLEU 评估
│
├── models/            # 网络结构
│   ├── __init__.py
│   ├── transformer.py # Encoder、Decoder、PE、Embedding、Generator
│   └── modules.py     # 多头注意力、前馈、残差层
│
├── utils/             # 工具组件
│   ├── __init__.py
│   ├── tokenizer.py   # 分词与编码
│   ├── dataset.py     # Dataset、Batch、mask
│   ├── loss_utils.py  # LabelSmoothing、NoamOpt
│   └── predict.py     # 贪婪解码与推理
│
├── sample/TM-training-set
│   ├── english.txt        # 英文语料
│   └── chinese.txt        # 中文语料
└── model/             # 训练中自动保存的权重与日志
```

