## Requirements
1. pipenv: To create Python virtual environment.
2. Python 3.7
3. The required Python packages are listed under "./Pipfile".
## Installation and Run
1. Create the pipenv virtual environment through 'pipenv shell'.
2. Install the packages through 'pipenv install'.
3. Run the code through 'python3 main.py'
## Project Structure
main.py : Entry point of this project. This file imports the prediction models and run the experiments.
data_prep.py : Code to preprocess the raw dataset. The processed dataset is formatted in the Dataset object.
group_finder.py : Code to find the groups of snips. This code is used only in data_prep.py.
estimator.py : Implementations of baseline prediction models.
autoencoder.py : Implementation of semi-supervised autoencoder for dynamic data.
utils.py : Code to conduct the experiments. The Experiment object will take (1) list of Estimators (2) Dataset object. Then Experiment object will conduct the prediction task on Dataset object using Estimators given.
