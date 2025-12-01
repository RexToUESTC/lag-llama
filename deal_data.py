import pandas as pd

data = pd.read_csv('energy_dataset.csv')

data=data[['time','price actual']]

data.to_csv('energy_dataset_dealed.csv')
