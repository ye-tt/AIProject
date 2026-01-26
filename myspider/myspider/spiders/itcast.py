import scrapy
from myspider.items import MyspiderItem
from bs4 import BeatifulSoup



'''
python -m scrapy startproject <project name> 
cd $projectfolder  相面的命令一定要在 项目里云运行
python -m scrapy genspider <爬虫名字> <允许爬虫的域名>
python -m scrapy crawl <爬虫名字>
'''

class ItcastSpider(scrapy.Spider):
    name = "itcast"
    allowed_domains = ["itcast.cn"]
    start_urls = ["https://www.itheima.com/teacher.html"] #Spider 会自动封装start url 到一个request object 

    def parse(self, response):
        #定义对于网站的相关操作，通常用于start url 对应的响应的解析

        # 获取所有教师节点
        teacher_nodes=response.xpath('//div[@class="swiper-slide"]')
        print(len(teacher_nodes))
        i=0

        for node in teacher_nodes :
            # item={}
            item = MyspiderItem()
            i +=1
            if(i==1):
                continue
            item['name']=node.xpath('.//h3/text()').extract()
            item['title']=node.xpath('.//span/text()').extract()
            # print(tmp['name'])
            # print(tmp['title'])
            yield item  ##每一次yield 都会执行MyspiderPipeline.process_item