You may have to manually isntall numba for UMAP to run through

```
pip install -U numba
```

```
python -m spacy download en_core_web_sm
```

Also don't forget to setup an environment file which looks similar to this:

```
 EN_CORPUS="/home/david/_MasterThesis/data/corpus/english/training-monolingual/news.2007.en.shuffled.sample"
 SEMCOR_CORPUS="/home/david/_MasterThesis/data/semcor-corpus/semcor/semcor/"
 TOP_20000_EN="/home/david/_MasterThesis/data/most_common/20k.txt"
```