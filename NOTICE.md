# Third-party data and code

## Spider dataset

This repository includes material from the Spider text-to-SQL dataset:

> Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li,
> Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. "Spider: A Large-Scale
> Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task."
> In *Proceedings of EMNLP 2018*.

- Source: <https://yale-lily.github.io/spider>
- License: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)

The following files contain Spider material and are distributed under CC BY-SA 4.0:

| path | relationship to Spider |
|---|---|
| `data/validation.json` | The Spider dev split, as distributed by the `premai-io/spider` mirror on Hugging Face |
| `cases.yaml` | Adapted: a stratified sample of 150 dev cases, with exclusions and adjudication annotations |
| `runs/` | Adapted: reproduces Spider questions, gold SQL, and database values alongside model output |

The Spider authors do not endorse this project. The SQLite databases are not stored in this
repository; `download_data.py` fetches them.

## Spider evaluation code

`vendor/spider_eval/` is from the Spider evaluation code and is licensed under the Apache
License 2.0. See `vendor/spider_eval/LICENSE`. `hardness.py` is extracted from `evaluation.py`.
