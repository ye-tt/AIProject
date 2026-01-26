import scrapy


class A58tongchengSpider(scrapy.Spider):
    name = "58tongcheng"
    allowed_domains = ["58.com"]
    start_urls = ["https://quanguo.58.com/ershouche/"]

#     custom_settings = {
#         'DEFAULT_REQUEST_HEADERS': {
#         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
#         'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
#         'Accept-Encoding': 'gzip, deflate, br, zstd',
#         'Connection': 'keep-alive',
#         'Host': 'www.google.com',
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
#     }
# }

    def parse(self, response):
        #提取数据
        node_list = response.xpath('//*[@id="list"]/ul')
        print(len(node_list))
        #翻页
