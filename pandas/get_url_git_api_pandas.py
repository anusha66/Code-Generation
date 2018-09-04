import requests
import time
import pdb

from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = date(2017, 12, 1)
end_date = date(2018, 9, 1)

token = "3690b2d4b0ed487434b395cb9096ff7160ca6c7f"

f = open('url_list.txt', 'a')

for single_date in daterange(start_date, end_date):
        print(single_date.strftime("%Y-%m-%d"))

        page = 1
        url_list = []

        while page <= 10:

            #print(page)

            url = "https://api.github.com/search/code?q=pandas+extension:ipynb+in:file+created="+single_date.strftime("%Y-%m-%d")+"&page="+str(page)+"&per_page=100&sort=stars&order=desc"
            headers = {
                'accept': "application/vnd.github.v3.text-match+json",
                'authorization': "token "+token,
            }

            data_json = requests.get(url=url, headers=headers).json()

            var = data_json.get('items',-1)

            if (var == []):
                #print("empty")
                break
            elif (var == -1):
                #print("sleep")
                if token == "3690b2d4b0ed487434b395cb9096ff7160ca6c7f":
                    token = "4d32711a67a9c13f85e98468c8168ed505cd3983"
                else:
                    token = "3690b2d4b0ed487434b395cb9096ff7160ca6c7f"
                time.sleep(3)
            else:
                #print("data")
                total_items = len(data_json['items'])

                for i in range(total_items):
                    url_list.append(data_json['items'][i]['html_url'])

                page = page + 1

                pdb.set_trace()

        time.sleep(1.5)

        for item in url_list:
            f.write("%s\n" % item)

f.close()

