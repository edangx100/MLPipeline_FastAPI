# ML pipeline using GitHub and Heroku

This project develops a classification model on publicly available Census Bureau data. Unit tests to monitor the model performance on various slices of the data were prepared. The model was deployed using the FastAPI package and API tests were prepared. Both the slice-validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.

## Procedures

### Data cleaning 

Execute `python main.py --action clean_data`

### Train/test model 

Execute `python main.py --action train_test_model`

### Check model slice score 

Execute `python main.py --action slice_score`

### Run all actions in pipeline

Execute `python main.py --action all` or `python main.py`

### Serve API locally

Execute `uvicorn server:app --reload`

### Check API deployed at Heroku

Execute `python check_heroku_api.py`

## CI/CD

At each git push, [Github workflow](.github/workflows/test_n_pulldata.yml) is run:
* Pulling of data from DVC is tested, 
* Flake8 is run to check code base against coding style (PEP8).
* Pytest is run to run unit tests.


## Files required for Rubric

* [dvcdag.png](screenshots/dvcdag.png)

* [slice_output.txt](screenshots/slice_output.txt)

* [Model Card](model_card.md)

* [example1.png](screenshots/example1.png)
* [example2.png](screenshots/example2.png)

* [live_get.png](screenshots/live_get.png)
* [live_post.png](screenshots/live_post.png)