# Machine Learning Project : Finding the Higgs Boson

In this project you will find : 
* an implementation library of some of the most commonly used supervised Machine Learning algorithms. [implementations.py](./scripts/implementations.py)
* a Jupyter notebook and run script [notebook](./scripts/project1.ipynb), [run](./scripts/run.py) using the before mentioned library in order to search a good model to classify particles from the dataset [train.csv](./data/train.csv),[test.csv](./data/test.csv) in order to determine if an observed particle is a Higgs Boson.


Note : proj1_helpers.predict_labels has been modified because we normalize all features and labels so our model is trained to predict values in [0 , 1]  predict_labels had to be modified and now put -1 where _[i]<=0.5, and 1 everywhere else.  