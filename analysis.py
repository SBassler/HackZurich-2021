##
import pandas as pd
import time
from pprint import pprint
import requests

##
df = pd.read_csv("daily_summaries.csv")
for uid in df.userId.unique():
    pass

##
df
##
x = df[df.userId == uid].activityMinutes
x
df.date = pd.to_datetime(df.date)
df["week_number"] = (
    df.groupby("userId").date.diff().dt.days.cumsum().fillna(0) // 7
)  # get the week number by the first
df
l = []
w = []
for wn in df.week_number.unique():
    x = df[df.week_number == wn].calories.to_numpy()
    l.append(x)
    w.append(wn)

import functools
import operator

data = []
for i in range(len(l)):
    x = l[i]
    wn = w[i]
    for xx in x.tolist():
        data.append((xx, wn))

##
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create the data
rs = np.random.RandomState(1979)
x = rs.randn(500)
# df = pd.DataFrame(dict(x=x, g=g))
df = pd.DataFrame(data, columns=('x', 'g'))
df = df[df.g > -8]
df['g'] = df['g'].apply(lambda x: f'{-x} week{"" if x == -1 else "s"} ago')
# m = df.g.map(ord)
# df["x"] += m

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=0.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=0.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(
        0,
        0.2,
        label,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


g.map(label, "x")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-0.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
import matplotlib.pyplot as plt
plt.show()
##
plt.hist(df[df.userId == uid].steps)
plt.show()

##
