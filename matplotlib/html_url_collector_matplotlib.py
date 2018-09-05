import requests
# from github import Github
import os
import time


# def scrape_data():
#     git = Github(login_or_token='6259f32c2eb35d19140b4cd04c78233b2c99a691')
#     s = git.search_code(query="matplotlib+extension:ipynb&created=2018-09-01", sort='indexed', order='desc')
#     print('bla')


days = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
days.extend([str(i) for i in range(10, 32)])

num_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

fp = open('pandas.txt', 'a')


def get_data():

    for i, month in enumerate(['09']):
        for j in range(30):
            for page in range(10):
                url = "https://api.github.com/search/code?q=matplotlib+extension:ipynb+created=2017-"+months[i]+"-"+days[j]+"&page="+str(page)+"&per_page=100&sort=stars&order=desc"
                headers = {
                    'Accept': 'application/vnd.github.v3.text-match+json',
                    'Authorization': "token 6259f32c2eb35d19140b4cd04c78233b2c99a691"

                    }
                while True:
                    res = requests.get(url=url, headers=headers).json()
                    if 'items' in res:
                        items = res['items']
                        break
                    else:
                        time.sleep(3)

                if len(items) == 0:
                    break

                for item in items:
                    fp.write(item['html_url'])
                    fp.write('\n')

            time.sleep(1.5)
        break
    fp.close()


def main():
    get_data()
    # scrape_data()

if __name__ == '__main__':
    main()
