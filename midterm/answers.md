# Mid Take Home Exam — 20210673 김대호

## 1. Dictionary-based sentiment analysis

The first 3,000 reviews are used without Q2 preprocessing. Positive is encoded as 1 and negative as 0.

| Question | Answer |
| --- | ---: |
| (1)(a) Number of positive reviews | 1,508 |
| (1)(b) Percentage of negative reviews | 1,492 / 3,000 × 100 = **49.7333%** |
| (2)(a) Mean AFINN score | **5.7466666667** |
| (2)(b) AFINN classification accuracy | 2,163 / 3,000 = **72.1000%** |
| (3)(a) Mean VADER compound score | **0.2921087333** |
| (3)(b) VADER classification accuracy | 2,107 / 3,000 = **70.2333%** |
| (4)(a) Mean TextBlob polarity | **0.1016683911** |
| (4)(b) TextBlob classification accuracy | 2,066 / 3,000 = **68.8667%** |

AFINN and TextBlob predict 1 only when their respective scores are greater than zero. VADER predicts 1 only when its positive score exceeds its negative score. All ties are assigned 0.

Confusion matrices have actual classes in rows and predicted classes in columns, both ordered [negative, positive]:

```text
AFINN          VADER          TextBlob
[[885, 607],   [[810, 682],   [[641, 851],
 [230,1278]]    [211,1297]]    [ 83,1425]]
```

## 2. Logistic regression

HTML tags are removed using BeautifulSoup with html5lib. Non-English-alphabetic characters are replaced with spaces, text is lowercased, NLTK English stopwords are removed, and tokens are joined with spaces. No stemming or lemmatization is added.

Reviews 1–2,100 are used for training and reviews 2,101–3,000 for testing, without shuffling. Both vectorizers use default parameters and are fitted only on the training reviews. LogisticRegression uses the lecture's default settings (scikit-learn 1.6.1). Each model has 25,267 input features.

| Question | Model | Correct / test reviews | Test accuracy |
| --- | --- | ---: | ---: |
| (1) | CountVectorizer + LogisticRegression | 748 / 900 | **83.1111%** |
| (2) | TfidfVectorizer + LogisticRegression | 756 / 900 | **84.0000%** |

TF-IDF correctly classifies eight more test reviews, improving accuracy by 0.8889 percentage points. Q1's scores evaluate all 3,000 reviews directly, while Q2's scores evaluate only the 900 held-out reviews.

Full executable code, confusion matrices, classification reports, and package versions are included in `midterm_20210673김대호.ipynb`. The scripts are `mid_Q1.py` and `mid_Q2.py`; detailed scores and predictions are in `results/`.
