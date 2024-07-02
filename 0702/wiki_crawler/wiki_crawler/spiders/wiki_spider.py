import scrapy

class WikiSpider(scrapy.Spider):
    name = "wiki_spider"
    allowed_domains = ["ko.wikipedia.org"]
    start_urls = [
        'https://ko.wikipedia.org/wiki/%ED%8E%98%EC%9D%B4%EC%BB%A4_(%ED%94%84%EB%A1%9C%EA%B2%8C%EC%9D%B4%EB%A8%B8)'
    ]

    def parse(self, response):
        infobox = response.css('#mw-content-text > div.mw-content-ltr.mw-parser-output > table.infobox.biography.vcard.vevent')

        rows = infobox.css('tr')
        infobox_data = {}
        
        for row in rows:
            header = row.css('th::text').get()
            data = row.css('td::text').get()
            if header and data:
                infobox_data[header.strip()] = data.strip()
            elif header and not data:
                infobox_data[header.strip()] = ''
            elif not header and data:
                infobox_data[''] = data.strip()

        yield infobox_data
