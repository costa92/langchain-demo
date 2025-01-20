# 导入所需的库
import json
import requests
import time

# 定义爬取微博用户信息的函数
def scrape_weibo(url: str):
    '''爬取相关鲜花服务商的资料'''
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
        "Referer": "https://weibo.com"
    }
    cookies = {
    	"cookie": '''SUB=_2AkMREKZ9f8NxqwFRmfwUyWzia4l0ygzEieKnTFemJRMxHRl-yT8Xqk4ttRB6OpCIkrobDN6QbOS6v0pqpJgbtHvUEClK; XSRF-TOKEN=Sjzpqd8mO4Bdg9y2wgSG2nxH; WBPSESS=Wk6CxkYDejV3DDBcnx2LOUPeEVdGvEkChS0EjxvVhTxR87NbsAmmo7giuxbbEe2e9UUQMcZHb9lhzO6XAYZFBQUvtLBaSRQ8Cji5L16LoWxexnOkNC7BiNFV3kP09atFk8RESLZ7ZlXBqm-WBe5dEu-7lThDJhE0wCZmQb5kU9A='''
    }
    response = requests.get(url, headers=headers, cookies=cookies)
    time.sleep(3)   # 加上3s 的延时防止被反爬
    return response.text

# 根据UID构建URL爬取信息
def get_data(id):
    url = "https://weibo.com/ajax/profile/info?uid={}".format(id)
    print(url)
    html = scrape_weibo(url)
    print(html)
    response = json.loads(html)

    return response