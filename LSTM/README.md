# LSTM Training and Prediction of Hardware Usage in Server Park

## Usage
### Define 
machines, criterion, time range, network hyperparameters at settings.py
OBS: soft_dtw_cuda.SoftDTW here is used an option of criterion (loss) since it is a differentiable version of DTW distance.

### Run 
```
python -m train_and_predict 
```
for running full train and prediction and saving hyperparameters file, results and output images using LSTM network and settings defined in settings.py

### Run
```
python -m analyse_results
```
for obtaining dataframe with summarization of results obtained from running train_and_predict in multiple settings

### Run 
```
python -m script_att
```
for running full train and prediction and saving hyperparameters file, results and output images using LSTM model with Attention Mechanism (decoder can also be specified in the same script - other settings are imported from settings.py) - implemented in different framework
