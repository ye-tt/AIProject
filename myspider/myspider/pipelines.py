# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json

class MyspiderPipeline:
    def __init__(self):
        self.file = open('itcast.json','w', encoding='utf-8')

    def process_item(self, item, spider):
        print(item)
        item = dict(item) # 将item 对象转化成 dict
        json_str = json.dumps(item, ensure_ascii=False, indent=2)
        self.file.write(json_str)
        ##默认使用完管道后需要将数据返回给引擎
        return item

    def __close__(self):
        self.file.close()