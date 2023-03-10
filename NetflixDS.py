# dataset -> https://www.kaggle.com/datasets/senapatirajesh/netflix-tv-shows-and-movies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

df = pd.read_csv("https://www.kaggle.com/datasets/senapatirajesh/netflix-tv-shows-and-movies/NetFlix.csv")
df.head()

df.shape

df.info()

df.describe()

df.columns
df['cast'].unique()
df['title'].unique()

#barplot
sns.barplot(df['release_year'],df['genres'][:10])
sns.barplot(df['duration'],df['genres'][:10])
#lineplot
sns.lineplot(df['release_year'], df['duration'])

#pie chart
df['genres'][:15].value_counts().plot(kind='pie',)
#pairplot
sns.pairplot(df)
