'''
Demo script
'''
from spatialaudiometrics import load_data as ld

# Firstly load your data (here we will load an example csv from the package but feel free to us pd.read_csv('filename'))
df = ld.load_example_behavioural_data()

df = ld.preprocess_behavioural_data(df)
kaja = []