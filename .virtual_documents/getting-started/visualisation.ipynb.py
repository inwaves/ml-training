import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/diabetes-regression.csv")

df.head()


fig, ax = plt.subplots(2, 1)
ax[0].plot(df['AGE'], df['Y'], color='r', marker='o')
ax[1].plot(df['BMI'], df['Y'], color='b', marker='o')
sns.despine()
plt.show()



