#
# import requests
# from bs4 import BeautifulSoup
# url='http://www.cntour.cn/'
# strhtml=requests.get(url)
# soup=BeautifulSoup(strhtml.text,'lxml')
# data=soup.select('#main>div>div.mtop.firstMod.clearfix>div.centerBox>ul.newsList>li>a')
# print(data)
#
# for item in data:
#     result={
#         'title':item.get_text(),
#         'link':item.get('href')
#     }
# print(result)
#
# import re
# for item in data:
#     result={
#         "title":item.get_text(),
#         "link":item.get('href'),
#         'ID':re.findall('\d+',item.get('href'))
#     }
# print(result)

from bs4 import BeautifulSoup
import requests

if __name__=="__main__":
    target='http://www.biqukan.com/1_1094/5403177.html'
    req=requests.get(url=target)
    html=req.text
    bf=BeautifulSoup(html,'lxml')
    texts=bf.find_all('div',class_='showtxt')
    print(texts[0])
    print(texts[0].text.replace('\xa0'*8,'\n\n'))

