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
| Pipeline 2 | `distilbert-base-uncased` / `distilroberta-base` | `oh-guest-segment` | business · leisure |
| Baselines | `nlptown/bert-base-multilingual-uncased-sentiment`, `facebook/bart-large-mnli` (zero-shot) | – | – |

## Repository layout

```
ISOM5240_GroupProject/
├── notebooks/                        # clean notebooks (open in Colab and Run all)
│   ├── 00_Data_Preparation.ipynb     # download, clean, label, balance, split, upload dataset
│   ├── 01_Finetune_Sentiment.ipynb   # Ou Zhiyong: fine-tune + select the sentiment model
│   ├── 02_Finetune_Segment.ipynb     # Zhang Ludan: fine-tune + select the segment model
│   └── 03_Experiments.ipynb          # model selection (accuracy, runtime) + app performance
├── notebooks_executed/               # the same notebooks with the outputs reported
├── app/
│   ├── app.py                        # Streamlit application
│   └── requirements.txt
├── results/Experimental_results.xlsx # all experiment results
├── report/                           # project report (PDF / Word) and figures
└── presentation/                     # slides and video script
```

**Team (Group 6):** Ou Zhiyong (A0018498R) · Zhang Ludan (A0309855R)
**App:** <https://oriental-horizon-review.streamlit.app/>

## Dataset

[515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)
(Booking.com reviews, public domain). Labels come from the data itself:

* **Sentiment:** reviewer score below 5.5 = negative, 6.5–7.9 = neutral, 9.0 and above = positive
  (the gaps remove ambiguous scores).
* **Segment:** trip tags `Business trip` = business, `Leisure trip` = leisure.
  A first version with three segments (business / couple / family) reached only 55.8% test
  accuracy, because most reviews do not reveal who the guest travelled with. It was replaced by
  the business-vs-leisure split, which is also the core segmentation in hotel revenue management.

Each task gets a class-balanced sample of up to 15,000 reviews per label, split 80/10/10 (seed 42).

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

**Reproducibility.** All random seeds are fixed (42), so the data sample and splits are identical
on every run. Re-running the GPU fine-tuning can move accuracy by a few tenths of a percentage
point, because some GPU operations are not deterministic. The models used by the app and reported
in our results are the ones uploaded to the Hugging Face Hub.
