Code runs on Python 3.7.8 (Clang 11.0.3)

Use 'pip_requirements.txt' to load required libraries:

```
pip install -r pip_requirements.txt
```

NLTK stopwords list is included in the project

To install word embeddings files, uncomment lines 173-175 in 'main.py'. Uncompress downloads and copy files to 'Vectors' folder

'enchant' library is include in 'pip_requirements.txt' but underlying library should be installed on OS:

e.g on Mac OS:
```
brew install libomp
install brew update & brew install enchant
```

Run 'python main.py' to run all models


