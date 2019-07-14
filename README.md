# Cross-Lingual Text Classification using Muse Embeddings and Deeplearning

## Information
<p>
In this project, text-classification model (sentiment analysis) is trained by using Facebook's Muse Embeddings (English). The same model can be used to classify the text having different language without requiring machine translation to english or retraining.
</p>

### Models Trained
1. Simple Dense Network,  Input - > Average of Token Embeddings
2. LSTM Network, Input -> Document Embeddings as a Sequence


## Results
#### Model Trained on - English<br>

Method 1 - Vector Average
1. German - 70.5
2. French - 72.45

Method 2 - LSTM Network (Can be improved with fine-tuning input and network)
1. German - 70.15
2. French - 67.9

## Dataset
<p>Amazon Book review in English, French and German (Attached in the repo).</p>

* Training - 2000 records
* Testing - 2000 records
* Ratings are used for labelling the records


## Muse Embeddings
Download the English (en) French (fr) and German (de) embeddings:
```bash
# English MUSE embeddings
curl -o data/wiki.en.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec
# French MUSE Wikipedia embeddings
curl -o data/wiki.fr.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fr.vec
# German MUSE Wikipedia embeddings
curl -o data/wiki.de.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.de.vec
```

## Dependencies - Python 3.6
1. Tensorflow
2. Keras
3. scikit-learn
4. nltk
5. pandas