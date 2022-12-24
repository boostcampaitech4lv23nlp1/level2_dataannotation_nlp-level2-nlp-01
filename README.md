# **Naver Boostcamp AI Tech level 2 NLP 1조**

## **[NLP] 문장 내 게체간 관계 추출 데이터 제작**

한국어 및 다른 언어에서의 자연어처리 데이터셋의 유형 및 포맷이 어떤지,

그리고 데이터셋을 구축하는 프로세스를 구축합니다.

위키피디아에서 '곤충' 키워드에 대한 원시 말뭉치를 활용하여

직접 관계 추출 테스크에 쓰이는 주석 코퍼스를 제작하는 것이 이 프로젝트의 목표입니다.

<br><br><br>
## **Contributors**

|이름|id|역할|
|:--:|:--:|--|
|강혜빈|[@hyeb](https://github.com/hyeb)|가이드라인 작성|
|권현정|[@malinmalin2](https://github.com/malinmalin2)|Tagtog 플랫폼에 문장 올리기|
|백인진|[@eenzeenee](https://github.com/eenzeenee)|Relation map 작성|
|이용우|[@wooy0ng](https://github.com/wooy0ng)|IAA 계산 / 모델 튜닝|
|이준원|[@jun9603](https://github.com/jun9603)|가이드라인 작성|


<br><br><br>

## **Data**

제작한 데이터셋
- train set : 1,022개
- validation set : 129개
- evaluation set : 129개

✓ 팀에게 제공된 '곤충' 키워드 원시 데이터 약 1500개에 대해서 데이터셋을 제작하였음. 

✓ 데이터 검수와 IAA 테스트를 마쳤음.

||raters|Fleiss's Kappa|
|:--:|--|--|
|2차 파일럿 태깅 후|5|0.731|
|최종 파일럿 태깅 후|5|0.815|


<br><br><br>

## **Time Line**

|Date|TODO|  
|:--:|--|
|'22.12.08 ~ '22.12.09|초기 가이드라인 작성 & 관계 탐색|
|'22.12.10 ~ '22.12.11|(1차) 파일럿 태깅 수행|
|'22.12.12 ~ '22.12.12|(1차) 가이드라인 작성 & 관계 탐색|
|'22.12.12 ~ '22.12.12|(2차) 파일럿 태깅 수행|
|'22.12.13 ~ '22.12.14|(2차) 가이드라인 작성 & 관계 탐색|
|'22.12.13 ~ '22.12.14|IAA 계산|
|'22.12.15 ~ '22.12.15|모델링|
|'22.12.15 ~ '22.12.15|EDA|




<br><br><br>

## **Evaluation Score**

|metrics|value|  
|:--:|--|
|accuracy|0.84|
|micro_f1|85.86|
|auprc|85.08|





<br><br><br>

## **project tree**

```
├── README.md
├── calculate_iaa.py
├── dataset
│   ├── test
│   │   └── evaluation.csv
│   └── train
│       ├── train.csv
│       └── validation.csv
├── fleiss.py
├── iaa_data
│   ├── iaa_sample.xlsx
│   └── prev_iaa_data.xlsx
└── template
    ├── dataloader.py
    ├── dict_label_to_num.pkl
    ├── dict_num_to_label.pkl
    ├── inference.py
    ├── metrics.py
    ├── models.py
    ├── requirements.txt
    └── train.py
```

<br><br><br>

## **Train**

```
$ python main.py --augment [value]
```

### **augment**
- `--tokenizer_name` : huggingface tokenizer name (str)
- `--model_name` : huggingface model name (str)
- `--batch_size` : batch_size (int)
- `--max_epoch` : epoch_size (int)
- `--learning_rate` : learning rate (float)
- `--train_path` : train dataset's path (str)
- `--dev_path` : validation dataset's path (str)
- `--test_path` : evaluation dataset's path (str)
- `--predict_path` : prediction dataset's path (str)

<br><br><br>

## **Inference**

```
$ python inference.py --augment [value]
```

### **augment**
- `--tokenizer_name` : huggingface tokenizer name (str)
- `--model_name` : huggingface model name (str)
- `--predict_path` : prediction dataset's path (str)


<br><br><br>


