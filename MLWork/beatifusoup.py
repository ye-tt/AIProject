from bs4 import BeautifulSoup
'''
<html>
 <head>
  <title>
   BeautifulSoup技术
  </title>
 </head>
 <body>
  <p class="title">
   <b>
    静夜思
   </b>
  </p>
  <p class="content">
   窗前明月光，
   <br/>
   疑似地上霜。
   <br/>
   举头望明月，
   <br/>
   低头思故乡。
   <br/>
  </p>
  <p class="other">
   李白（701年－762年），字太白，号青莲居士，又号“谪仙人”，
    唐代伟大的浪漫主义诗人，被后人誉为“诗仙”，与
   <a class="poet" href="http://example.com/dufu" id="link1">
    杜甫
   </a>
   并称为“李杜”，为了与另两位诗人
   <a class="poet" href="http://example.com/lishangyin" id="link2">
    李商隐
   </a>
   、
   <a class="poet" href="http://example.com/dumu" id="link3">
    杜牧
   </a>
   即“小李杜”区别，杜甫与李白又合称“大李杜”。
    其人爽朗大方，爱饮酒...
  </p>
  <p class="story">
   ...
  </p>
 </body>
</html>
'''
soup = BeautifulSoup(open('../智玛AI课程/test.html', 'r', encoding='utf-8'), 'lxml')
'''
1. 定位节点
    soup.tag 拿到根据标签名字拿到节点, 只找到第一个符合条件的节点
    find(tag) 返回一个对象
    find(tag, attributename=value)
    find_all(tag) 返回一个列表
    select  推荐用select 函数
    select(element)
    select(.classname)
    select(#id)
    select(属性选择器)
'''
title =soup.title

a1=soup.find('a')

##Name
# print(a1.name)
#content
# print(a1.string)
##Attributes
# print(a1.attrs)
# print(a1['class'])
# print(a1.get('id'))
# print(title)
# print(title.name) 
# print(title.attrs)


#修改属性 增加属性name
a1['class'] = 'abc'
a1['id'] = '1'
a1['name'] = '2'
# print(a1)

##删除属性
# del a1['class']
# del a1['name']
# print(a1)
# print(a1['class'])  will show error, no class attribute

a2=soup.find('a', id='link1')
# print(a1)
# print(a2)
a_list= soup.find_all('a')

#从文档中找到<a>的所有标签链接
for a in soup.find_all('a'):
    # print(a)
    pass
# print(a_list)

#获取<a>的超链接
for link in soup.find_all('a'):  
    # print(link.get('href'))
    pass

#获取<a>的文本内容
for a in soup.find_all('a'):
    # print (a.get_text())
    pass

title_a_list= soup.find_all(['title','a'])## 返回所有的Title 和 a 的节点

first_a = soup.select('a') #select(element)
# print(first_a)
content_p=soup.select('.content') #select(.classname)
# print(content_p)
a_link1=soup.select('#link1') #select(#id)
# print(a_link1)

##单独获取某个属性值
a_class_name = soup.a['class']
# print(a_class_name)
a_class_name = soup.a.get('class')

##层级选择

a_new = soup.select('p a')
# print(a_new)

#### 层级选择
a_link3 = soup.select('p a#link3') 
p_title = soup.select('body p.title') 

print(p_title)