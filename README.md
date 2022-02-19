# ML pipeline using GitHub and Heroku

This project develops a classification model on publicly available Census Bureau data. Unit tests to monitor the model performance on various slices of the data were prepared. The model was deployed using the FastAPI package and API tests were prepared. Both the slice-validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.

## Code testing

Execute `pytest` to run tests.

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

Execute `uvicorn api_server:app --reload`

### Check API deployed at Heroku

Execute `python check_heroku_api.py`

## CI/CD

Github workflow [Test pipeline](.github/workflows/test_n_pulldata.yaml) is triggered at each git push.
Pipeline test pulling of data from DVC, execute Flake8 + pytest doing every test.


## Files required for Rubric

* [dvcdag.png](screenshots/dvcdag.png)

* [slice_output.txt](screenshots/slice_output.txt)

* [Model Card](model_card.md)

* [example1.png](screenshots/example1.png)
* [example2.png](screenshots/example2.png)

* [live_get.png](screenshots/live_get.png)
* [live_post.png](screenshots/live_post.png)