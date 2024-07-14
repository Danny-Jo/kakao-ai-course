# @title
# wikipedia_scraper/spiders/wikipedia_spider.py
import scrapy

class WikipediaSpider(scrapy.Spider):
    name = "wikipedia_spider"
    start_urls = [
        'https://ko.wikipedia.org/wiki/위키백과:대문',
    ]

    def parse(self, response):
        main_content = response.css('#mw-content-text > div.mw-content-ltr.mw-parser-output > div.main-pane > div.main-pane-right > div.wikipedia-ko.main-recommended.main-box').get()
        yield {
            'main_content': main_content,
        }
