#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:39:33 2017

@author: Lesile
"""

"""
###Chapter14 Web集成
用于Web的Python是一个宽泛的主题，本章主要介绍金融方面的重要主题:
    Web协议:第一小节说明如何通过FTP传输文件,通过HTTP访问网站
    Web图表绘制:Bokeh
    Web应用:Flask
    Web服务
"""

"""
###14.1 Web基础知识
ftplib
httplib
urllib
"""
### ftplib
import ftplib
import numpy as np
ftp = ftplib.FTP('YOUR_SERVER_DOMAIN.com')
ftp.login(user='REPLACE', passwd='REPLACE')
np.save('./data/array', np.random.standard_normal((100, 100)))
f = open('./data/array.npy', 'r')
ftp.storbinary('STOR array.npy', f)
ftp.retrlines('LIST')
f = open('./data/array_ftp.npy', 'wb').write
ftp.retrbinary('RETR array.npy', f)
ftp.delete('array.npy')
ftp.retrlines('LIST')
ftp.close()

###httplib
import http
import http.client
http = http.client.HTTPConnection('hilpisch.com')
http.request('GET', '/index.htm')
#使用getresponse方法测试请求是否成功
resp = http.getresponse()
resp.status, resp.reason
#读取内容
content = resp.read()
content[:100]
index = content.find(b' E ')
index
content[index:index + 29]
http.close()

###urllib
import urllib.request
url = 'http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv'
url += '&s=YHOO&a=01&b=1&c=2014&d=02&e=6&f=2014'
connect = urllib.request.urlopen(url)
data = connect.read()
print(data)

url = 'http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv'
url += '&%s'  # for replacement with parameters
url += '&d=06&e=30&f=2014'


params = urllib.parse.urlencode({'s': 'MSFT', 'a': '05', 'b': 1, 'c': 2014})
params

url % params
connect = urllib.request.urlopen(url % params)
data = connect.read()
print(data)

urllib.request.urlretrieve(url % params, './data/msft.csv')
csv = open('./data/msft.csv', 'r')
csv.readlines()[:5]




"""
###14.2 Web图表绘制
Bokeh
"""

"""
###14.3  快速Web应用
Flask
"""


"""
###14.4 Web服务
"""

