Code runs on Python 3.7.8 (Clang 11.0.3)

Use `pip_requirements.txt` to load required libraries:

```
pip install -r pip_requirements.txt
```

NLTK stopwords list is included in the project

To install word embeddings files, uncomment lines 173-175 in `ve/main.py` and run. Uncompress downloads and copy files to `ve/Vectors` folder

`enchant` library is include in `pip_requirements.txt` but underlying library should be installed on OS:

e.g on Mac OS:
```
brew install libomp
install brew update & brew install enchant
```

See https://pyenchant.github.io/pyenchant/install.html for other examples.

Dataset should be installed as a tab-seperated file in `ve` folder with with the name `founta-dataset.csv`. 

Code expects a header line. If necessary, tabs should be escaped with `"` 

Example first lines with the label `hateful`:

```
text	label confidence
This is some text	hateful 4
```

Run `python /ve/main.py` to run all models


