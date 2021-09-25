##
import pandas as pd
import time
from pprint import pprint
import requests
import json
from tqdm import tqdm

##
api_key = "e195eaa3-e9bd-4f82-9f83-ca6466cac6ed"
h = {
    "X-Tenant": "hackzurich",
    "Content-Type": "application/json",
    "X-API-Key": api_key,
}
all_j = []
users = []

##
class User:
    def __init__(self, user_id, first_name, last_name, profile_pic_url):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.profile_pic_url = profile_pic_url


##
# a = 0
# b = 1
a = 400
b = 600
for i in tqdm(range(a, b)):
    n = 0
    a = requests.get("https://api.fitrockr.com/v1/users?page=0&size=100", headers=h)
    # print(a.__dict__)
    js = a.content.decode("utf-8")
    j = json.loads(js)
    for jj in j:
        u = User(
            user_id=jj["id"],
            first_name=jj["firstName"],
            last_name=jj["lastName"],
            profile_pic_url=jj["profilePicUrl"],
        )
        n += 1
        users.append(u)
    print(f"found {n} users")
    time.sleep(2)
    # pprint(j)


def users_list_to_csv(users):
    l = [(u.user_id, u.first_name, u.last_name) for u in users]
    df = pd.DataFrame(l, columns=["user_id", "first_name", "last_name"])
    print(f"exported {len(l)} users to csv")
    df.to_csv("30k_users.csv", index=False, sep=",")


users_list_to_csv(users)
##
len(users)
##
len(users)

request = f"""\
GET /v1/status/greet HTTP/1.1
Content-Type: application/json
X-Tenant: example-tenant
X-API-Key: {api_key}
Host: api.fitrockr.com"""

##
import subprocess

s = (
    f'/usr/bin/curl https:/api.fitrockr.com/v1/users?page=0&size=10 -H "X-Tenant: hackzurich" -H "Content-Type: '
    f'application/json" -H '
    f'"X-API-Key: e195eaa3-e9bd-4f82-9f83-ca6466cac6ed"'
)

subprocess.check_output(s, shell=True)
