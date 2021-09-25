##
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df0 = pd.read_csv(
    "/Users/macbook/Dropbox/shared_folders/shared_files/hack_zurich2021/final_move_set.csv"
)
df1 = pd.read_csv(
    "/Users/macbook/Dropbox/shared_folders/shared_files/hack_zurich2021/final_profile_set.csv"
)

##
df0.drop(columns=["Unnamed: 0", "...1"], inplace=True)

##
df0["new_date"] = pd.to_datetime(df0.date, format="%Y-%m-%d")
##

##
for uid in df0.userId.unique():
    pass
##
df0["rolling_score"] = df0["score"].rolling(7).mean().shift(-3)
plt.figure(figsize=(10, 4))
df0u = df0[df0.userId == uid]
g = sns.lineplot(data=df0u, x="new_date", y="score", hue="recognized_activity")
g.legend(loc="center left", bbox_to_anchor=(1.25, 0.5), ncol=1)
# plt.fill_between(df0u.new_date.values, df0u.new_date.values)
plt.tight_layout()
plt.show()
df0.columns
df0.iloc[0]
##
l = []
for v in df0u.recognized_activity.unique():
    sub_df = df0u[df0u.recognized_activity == v]

##
"""
ADJUSTING DATETIME IN SLEEP DATA
"""
df = pd.read_csv("sleep.csv")
len(np.unique(df[["endTime", "userId"]].values))
pd.set_option("expand_frame_repr", False)


def f(x):
    x = x.replace("'", '"')
    import json

    x = json.loads(x)
    s = (
        f'{x["date"]["year"]}-{x["date"]["month"]:02d}-{x["date"]["day"]:02d} {x["time"]["hour"]:02d}:'
        f'{x["time"]["minute"]:02d}:{x["time"]["second"]:02d}'
    )
    # s = f'{x["date"]["year"]}-{x["date"]["month"]:02d}-{x["date"]["day"]:02d}'
    return s


df["newCalendarDate"] = df.processingDateTime.apply(f)
print(len(np.unique(df[["newCalendarDate", "userId"]].values)))
df.drop(
    columns=["sleepLevelsMap", "sleepEfficiency", "unmeasurableSeconds", "validation"],
    inplace=True,
)
df.columns
df
df.to_csv("sleep_revisited.csv")

##
for gr in df0.groupby(by=["recognized_activity", "userId"]):
    gr[1].score.hist()
    plt.title(gr[0])
    plt.show()

##
c0 = (26 / 255, 44 / 255, 77 / 255)
c1 = (130 / 255, 156 / 255, 203 / 255)
# df['id'].value_counts()
plt.figure(figsize=(10, 10))
df = pd.read_csv(
    "/Users/macbook/Dropbox/shared_folders/shared_files/hack_zurich2021/calculated_stuff/final_sleep_set.csv"
)
df = df[df.id == "613db206309a3f06bfa3d028"]
df = df.loc[df["recognized_activity"] != "duration_hours"]
# df[df['recognized_activity'] == 'duration_hours'].score
df["newCalendarDate"] = pd.to_datetime(df.newCalendarDate, format="%Y-%m-%d")
# df['score'] = df.score / 60
pd.crosstab(
    df.newCalendarDate, df.recognized_activity, values=df.score, aggfunc=sum
).plot.bar(stacked=True, color=(c0, c1))
ax = plt.gca()
xticks = ax.get_xticklabels()
for i in range(len(xticks)):
    # xticks[i].__dir__()
    t = xticks[i].get_text()
    xticks[i].set_text(t[:10])
ax.set_xticklabels(xticks)
ax.set(xlabel="")
ax.legend(labels=("deep sleep", "light sleep"))  # 'rem'
yticks = ax.get_yticklabels()
# for i in range(len(yticks)):
#     xticks[i].__dir__()
# t = yticks[i].get_text()
# yticks[i].set_text(f'{int(t) // 20 * 7}')
# ax.set_yticklabels(yticks)
#
plt.ylabel("hours")
#
df
plt.tight_layout()
plt.show()
# df.set_index('recognized_activity')[['newCalendarDate']]
# sns.set()
# dff = df.set_index('recognized_activity')[['newCalendarDate']]
# dff
# dff.plot(kind='bar', stacked=True)
# plt.show()

##
