# 笔趣网
# # import requests
# #
# # url='http://www.biqukan.com/1_1094/5403177.html'
# # reques=requests.get(url)
# # print(reques.text)
#
# # from bs4 import BeautifulSoup
# # import requests
# #
# # if __name__=="__main__":
# #     target='http://www.biqukan.com/1_1094/5403177.html'
# #     req=requests.get(url=target)
# #     html=req.text
# #     bf=BeautifulSoup(html)
# #     texts=bf.find_all('div',class_='showtxt')
# #     print(texts)
# #     print(texts[0].text)
# #     # print(texts[0].text.replace('\xa0'*8,'\n\n'))
#
#
from bs4 import BeautifulSoup
import requests
if __name__ == "__main__":
     server = 'http://www.biqukan.com/'
     target = 'http://www.biqukan.com/1_1094/'
     req = requests.get(url=target)
     # req.encoding = 'GB2312'#解决乱码
     req.encoding=req.apparent_encoding
     html = req.text
     div_bf = BeautifulSoup(html,'lxml')
     div = div_bf.find_all('div', class_ = 'listmain')
     a_bf = BeautifulSoup(str(div[0]),'lxml')
     a = a_bf.find_all('a')
     print(a[13:])
     for each in a:
          print(each.string, server + each.get('href'))






