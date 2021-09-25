##
import pandas as pd
import pandas as pd
import time
from pprint import pprint
import requests
import json
from tqdm import tqdm

##
l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
##
users = pd.read_csv("3700_users.csv")
len(users)

##
api_key = "e195eaa3-e9bd-4f82-9f83-ca6466cac6ed"
h = {
    "X-Tenant": "hackzurich",
    "Content-Type": "application/json",
    "X-API-Key": api_key,
}
##
# date_range = 'startDate=2020-07-20&endDate=2021-09-25'
date_range = "startDate=2021-09-10&endDate=2021-09-25"


def user_request(s):
    a = requests.get(f"https://api.fitrockr.com/v1/users/" + s, headers=h)
    js = a.content.decode("utf-8")
    j = json.loads(js)
    time.sleep(0.9)
    return j


##
for i in range(3):
    uid = users.iloc[i]["user_id"]
    # pprint(user_request(f"{uid}/profile"))
    # pprint(user_request(f'{uid}/dailySummaries?{date_range}'))
    # pprint(user_request(f'{uid}/dailyDetails?{date_range}')[0])
    # pprint(user_request(f'{uid}/heartRate?{date_range}')[0:2])
    # pprint(user_request(f'{uid}/motionIntensity?{date_range}'))
    pprint(user_request(f"{uid}/sleep?{date_range}"))
    time.sleep(1)

##
def save_dfs():
    df0 = pd.DataFrame.from_dict(l0, orient="columns")
    df0.to_csv("user_profile.csv")
    print(f"saved {len(df0)} rows to user_profile.csv")

    df1 = pd.DataFrame.from_dict(l1, orient="columns")

    def f(x):
        x = json.loads(x.replace("'", '"'))
        return g(x)

    def g(x):
        return f'{x["year"]}-{x["month"]:02d}-{x["day"]:02d}'

    df1["date"] = df1["date"].apply(g)
    df1.to_csv("daily_summaries.csv")
    print(f"saved {len(df1)} rows to daily_summaries.csv")

    df2 = pd.DataFrame.from_dict(l2, orient="columns")
    df2.to_csv("sleep.csv")
    print(f"saved {len(df2)} rows to sleep.csv")

    df3 = pd.DataFrame.from_dict(l3, orient="columns")
    df3.to_csv("daily_details.csv")
    print(f"saved {len(df3)} rows to daily_details.csv")

    d = dict(zip(list(range(len(l4))), l4))
    df4 = pd.DataFrame.from_dict(d, orient="columns").T
    df4.to_csv("heart.csv")
    print(f"saved {len(df4)} rows to heart.csv")


# save_dfs()
##

a = 0
b = 300
for ii, i in enumerate(tqdm(range(a, b))):
    uid = users.iloc[i]["user_id"]

    j0 = user_request(f"{uid}/profile")
    l0.append(j0)

    j1 = user_request(f"{uid}/dailySummaries?{date_range}")
    l1.extend(j1)

    j2 = user_request(f"{uid}/sleep?{date_range}")
    l2.extend(j2)

    j3 = user_request(f"{uid}/dailyDetails?{date_range}")
    l3.extend(j3)

    j4 = user_request(f"{uid}/heartRate?{date_range}")
    l4.extend(j4)

    if ii % 50 == 0:
        save_dfs()

save_dfs()

##
df["value"].value_counts()


df = pd.read_csv("user_profile.csv")
df.imperialUnits.to_numpy().sum()

##
df = pd.read_csv("daily_summaries.csv")
d = df["date"][0]
print(d)

##
df = pd.read_csv("sleep.csv")
df.iloc[0]
df.iloc[0].to_dict()["sleepLevelsMap"]
df.columns

##
pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
df = pd.read_csv("sleep.csv")
pprint(df)
pd.set_option("expand_frame_repr", True)

##
len(set(df.userId.tolist()))
