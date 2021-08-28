import pandas as pd
import numpy as np

data = pd.read_csv (r'IRIS.csv')
iris = np.array(data)
dataset = iris.tolist();
trained_mob = [row[:-2] for row in dataset]
print(trained_mob)