import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stemgraphic
import seaborn as sns
from scipy.special import comb  # For combinatorial calculations (see line 209)
from scipy.stats import hypergeom # For dhyper() equivalent (see line 225)
from scipy.stats import binom # For dbinom() equivalent (see line 264)
from scipy.stats import poisson # For dpois() equivalent (see line 304)
from scipy.stats import nbinom # For dnbinom() equivalent (see line 324)
from scipy.stats import geom # For dgeom() equivalent (see line 338)
from scipy.stats import norm # For pnorm() equivalent (see line 349)
from scipy.stats import probplot # For qqnorm() equivalent (see line 367)
from scipy.stats import lognorm # For plnorm() equivalent (see line 429)
from scipy.stats import expon # For pexp() equivalent (see line 438)
from scipy.stats import gamma # For pgamma() equivalent (see line 453)
from scipy.stats import weibull_min # For pweibull() equivalent (see line 462)

df = pd.read_csv('/Users/abhijitghosh/Documents/DataScience/IN_chemistry.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, nutrient in zip(axes, ['NO3_epi', 'NH3_epi', 'Total_Phos_epi']):
    sns.scatterplot(x=df[nutrient], y=df['Secchi'], ax=ax)
    ax.set_title(f'Secchi Depth vs {nutrient}')
    ax.set_xlabel(f'{nutrient} (mg/L)')
    ax.set_ylabel('Secchi Depth (m)')

plt.show()

summary_table_1 = df[['NO3_epi', 'NH3_epi', 'Total_Phos_epi', 'Secchi']].corr()
print(summary_table_1)

df['Month'] = pd.to_datetime(df['Date_Sampled']).dt.month
print(df['Month'])
df['Season'] = df['Month'].apply(lambda x: 'Spring' if x in [3, 4, 5] else 
                                            'Summer' if x in [6, 7, 8] else 
                                            'Fall' if x in [9, 10, 11] else 'Winter')

plt.figure(figsize=(8, 5))
sns.boxplot(x='Season', y='Chlorophyll_a', data=df)
plt.title('Chlorophyll-a Concentration by Season')
plt.xlabel('Season')
plt.ylabel('Chlorophyll-a (µg/L)')
plt.show()

summary_table_2 = df[['Chlorophyll_a', 'Secchi', 'Total_Phos_epi', 'NH3_epi', 'NO3_epi']].corr()
print(summary_table_2)

plt.figure(figsize=(8, 5))
sns.histplot(df['Secchi'], bins=10)
plt.axvline(x=1.5, color='r', linestyle='--', label="Low Clarity Threshold (1.5m)")
plt.title('Histogram of Water Clarity (Secchi Depth)')
plt.xlabel('Secchi Depth (m)')
plt.ylabel('Frequency')
plt.show()

summary_table_3 = df[['Secchi', 'Total_Phos_epi', 'NH3_epi', 'NO3_epi']].describe().T[['mean', '50%', 'min', 'max', 'std']]
print(summary_table_3)




