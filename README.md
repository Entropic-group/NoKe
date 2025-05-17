# NoKe: A Fake News Detector Powered by Machine Learning 

This project is a machine learning application that detects fake news using a variety of models and techniques.

## Technologies and Concepts

* **Machine Learning models** including a baseline `Random Forest` implemented with `Scikit-learn` and a more advanced `RoBERTa` model built using HuggingFace Transformers` and `PyTorch Lightning`.
* **Natural Language Processing techniques**, including `TF-IDF vectorization` for traditional models and transformer-based representations for deep learning.
* **Custom feature engineering pipeline**, abstracted through dedicated featurizer classes that support training, caching, and serialization for consistent preprocessing.
* **Model evaluation** using standard classification metrics such as `F1 score, accuracy, AUC, and confusion matrix`, supported by the `scikit-learn metrics` suite.
* **Model interpretation and error analysis** enabled by `SHAP` (Shapley Additive Explanations) to provide insight into feature importance and decision boundaries.
* **Data version control** and pipeline reproducibility managed using `DVC` (Data Version Control).
* **Experiment tracking and configuration** handled through structured logging and `MLflow` integration.
* **Test infrastructure** including unit and integration tests using `PyTest and data validation with `Great Expectations`.
* **Deployment-ready backend** implemented using `FastAPI`, exposing RESTful endpoints.
* **`Docker` for deployment workflow, and `GitHub actions` for constinous integration (`CI`)**.
* **Web browser integration** through a custom `Chrome extension` that interfaces with the deployed model for direct content evaluation.

## Get started
This project uses [uv](https://docs.astral.sh/uv/) as project manager. To install it, follow the instruction [here](https://docs.astral.sh/uv/getting-started/installation/) for your operating system.

Once `uv` is installed, clone the repository:
```
git clone [REPO LINK HERE]
```

Now, activate the virtual environment and get all the dependencies installed: 
```
uv venv
uv sync
```

Get the data (training, testing, and valiation data) from [this repository](https://github.com/Tariq60/LIAR-PLUS/tree/master/dataset/tsv) and put it in `data/raw`.

### Train

To train the [random forest baseline](https://www.mihaileric.com/posts/machine-learning-project-model-v1/), run the following from the root directory:
```
dvc repro train-random-forest
```

Your output should look something like the following:
```
INFO - 2021-01-21 21:26:49,779 - features.py - Creating featurizer from scratch...
INFO - 2021-01-21 21:26:49,781 - tree_based.py - Initializing model from scratch...
INFO - 2021-01-21 21:26:49,781 - train.py - Training model...
INFO - 2021-01-21 21:26:50,163 - features.py - Saving featurizer to disk...
INFO - 2021-01-21 21:26:50,169 - tree_based.py - Featurizing data from scratch...
INFO - 2021-01-21 21:26:59,360 - tree_based.py - Saving model to disk...
INFO - 2021-01-21 21:26:59,459 - train.py - Evaluating model...
INFO - 2021-01-21 21:26:59,584 - train.py - Val metrics: {'val f1': 0.7587628865979381, 'val accuracy': 0.7266355140186916, 'val auc': 0.8156070164865074, 'val true negative': 381, 'val false negative': 116, 'val false positive': 235, 'val true positive': 552}
```

### Deploy

Hving trained the model, you can now deploy it. Build your deployment Docker image (this may take some minutes):
```
docker build . -f deploy/Dockerfile.serve -t fake-news-deploy
```

Now, run the model locally via a REST API with:
```
docker run -p 8000:80 -e MODEL_DIR="/home/fake-news/random_forest" -e MODULE_NAME="fake_news.server.main" fake-news-deploy
```

It should now be possible to interact with the API. For instance, you can do:
```
curl -X POST http://127.0.0.1:8000/api/predict-fakeness -d '{"text": "you can write some string here :)"}'
```
Tools like `Postman` can also be used.