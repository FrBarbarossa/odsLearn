import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# %config InlineBackend.figure_format = 'retina'


warnings.filterwarnings("ignore")

df = pd.read_csv('BigML_Dataset_1.csv')

features = ['total day minutes', 'total intl calls']
df[features].hist(figsize=(10, 4))
#
# df[features].plot(
#     kind="density", subplots=True, layout=(1, 2), sharex=False, figsize=(10, 4)
# )

# plt.plot(df[features])
plt.show()