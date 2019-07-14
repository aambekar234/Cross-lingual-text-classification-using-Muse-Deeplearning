# Cross-Lingual Text Classification using Muse Embeddings and Deeplearning

## Information
<p>
In this project, text-classification model is trained by using Facebook's Muse Embeddings (English). The same model can be used to classify the text having different language without requiring machine translation to english.
</p>

### Models Trained
1. Dense Network,  Input - > Average of Token Embeddings
2. LSTM Network


## Results
#### Model Trained on - English<br>

Method 1 - Vector Average
1. German - 70.5
2. French - 72.45

Method 2 - LSTM Network
1. German - 70.15
2. French - 67.9

## Dependencies - Python 3.6
1. Tensorflow
2. Keras
3. scikit-learn
4. nltk
5. pandas

## Dataset
<p>Amazon Book review in English, French and German (Attached in the repo).</p>

* Training - 2000 records
* Testing - 2000 records
* Ratings are used for labeling the records


## Muse Embeddings
You can download the English (en) French (es) and German (de) embeddings this way:
```bash
# English MUSE embeddings
curl -o data/wiki.en.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec
# French MUSE Wikipedia embeddings
curl -o data/wiki.fr.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fr.vec
# German MUSE Wikipedia embeddings
curl -o data/wiki.de.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.de.vec
```