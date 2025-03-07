# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import re
import webbrowser

from scrapy.conf import settings
from scrapy.exceptions import DropItem


class AirbnbScraperPipeline(object):
    
    def __init__(self):
        """Class constructor."""
        self._fields_to_check = ['description', 'name', 'summary', 'reviews']
        self._minimum_monthly_discount = int(settings.get('MINIMUM_MONTHLY_DISCOUNT', None))
        self._minimum_weekly_discount = int(settings.get('MINIMUM_WEEKLY_DISCOUNT', None))

        self._skip_list = settings.get('SKIP_LIST', None)

        self._cannot_have_regex = settings.get('CANNOT_HAVE', None)
        if self._cannot_have_regex:
            self._cannot_have_regex = re.compile(str(self._cannot_have_regex), re.IGNORECASE)

        self._must_have_regex = settings.get('MUST_HAVE', None)
        if self._must_have_regex:
            self._must_have_regex = re.compile(str(self._must_have_regex), re.IGNORECASE)

        self._web_browser = settings.get('WEB_BROWSER', None)
        if self._web_browser:
            self._web_browser += ' %s' # append URL placeholder (%s)
    
    def process_item(self, item, spider):
        """Drop items not fitting parameters. Open in browser if specified. Return accepted items."""

        if self._skip_list and str(item['id']) in self._skip_list:
            raise DropItem('Item in skip list: {}'.format(item['id']))

        if self._minimum_monthly_discount and 'monthly_discount' in item:
            if item['monthly_discount'] < self._minimum_monthly_discount:
                raise DropItem('Monthly discount too low: {}'.format(item['monthly_discount']))

        if self._minimum_weekly_discount and 'weekly_discount' in item:
            if item['weekly_discount'] < self._minimum_monthly_discount:
                raise DropItem('Weekly discount too low: {}'.format(item['weekly_discount']))

        # check regexes
        if self._cannot_have_regex:
            for f in self._fields_to_check:
                v = str(item[f].encode('ASCII', 'replace'))
                if self._cannot_have_regex.search(v):
                    raise DropItem('Found: {}'.format(self._cannot_have_regex.pattern))

        if self._must_have_regex:
            has_must_haves = False
            for f in self._fields_to_check:
                v = str(item[f].encode('ASCII', 'replace'))
                if self._must_have_regex.search(v):
                    has_must_haves = True
                    break

            if not has_must_haves:
                raise DropItem('Not Found: {}'.format(self._must_have_regex.pattern))

        # open in browser
        if self._web_browser:
            webbrowser.get(self._web_browser).open(item['url'])
            
        return item
