# Oriental Horizon · Review Intelligence for Revenue Management

ISOM5240 Group Project · **Oriental Horizon Hotel Business** (<https://orientalhorizon.com.cn/>)

**Objective:** This project fine-tunes deep-learning text classifiers that turn hotel guest
reviews into two pricing signals for the revenue management system: guest sentiment
(reputation) and guest segment (demand mix).

## Pipeline

```
guest reviews ──► Pipeline 1: fine-tuned sentiment model ──► Net Reputation Score ──► suggested rate adjustment
             └──► Pipeline 2: fine-tuned segment model   ──► segment mix          ──► segment pricing actions
```

| Component | Pre-trained model (Hugging Face) | Fine-tuned model | Labels |
|-----------|----------------------------------|------------------|--------|
| Pipeline 1 | `distilbert-base-uncased` / `distilroberta-base` | `oh-review-sentiment` | negative · neutral · positive |
| Pipeline 2 | `distilbert-base-uncased` / `distilroberta-base` | `oh-guest-segment` | business · couple · family |
| Baselines | `nlptown/bert-base-multilingual-uncased-sentiment`, `facebook/bart-large-mnli` (zero-shot) | – | – |

## Repository layout

```
ISOM5240_GroupProject/
├── notebooks/
│   ├── 00_Data_Preparation.ipynb     # download, clean, label, balance, split, upload dataset
│   ├── 01_Finetune_Sentiment.ipynb   # Student 1: fine-tune + select the sentiment model
│   ├── 02_Finetune_Segment.ipynb     # Student 2: fine-tune + select the segment model
│   └── 03_Experiments.ipynb          # model selection (accuracy, runtime) + app performance
└── app/
    ├── app.py                        # Streamlit application
    └── requirements.txt
```

## Dataset

[515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)
(Booking.com reviews, public domain). Labels come from the data itself:

* **Sentiment:** reviewer score below 5.5 = negative, 6.5–7.9 = neutral, 9.0 and above = positive
  (the gaps remove ambiguous scores).
* **Segment:** trip tags `Business trip` = business, `Leisure trip` + `Couple` = couple,
  `Leisure trip` + `Family with … children` = family.

Each task gets a class-balanced sample of 6,000 reviews per label, split 80/10/10
(train 14,400 / validation 1,800 / test 1,800, seed 42).

## How to reproduce

1. Open each notebook in Google Colab and choose a **T4 GPU** runtime for notebooks 01–03.
2. Optional, only for uploading: add a Hugging Face write token as the Colab secret `HF_TOKEN`.
3. Run the notebooks in order 00 → 01 → 02 → 03 with *Runtime → Run all*.
   Notebooks 01–03 load the dataset and models from the Hugging Face Hub, so they also run
   without a token.
4. Deploy `app/app.py` on Streamlit Community Cloud. The *Accuracy check* tab measures both
   models' accuracy on labelled test reviews on Streamlit Cloud.

Software: `transformers==5.16.0` in both the notebooks and the app, so the deployed models
behave the same as in the experiments.
