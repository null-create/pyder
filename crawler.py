import asyncio
from urllib.parse import urljoin
from typing import Callable, Dict, List, Optional, Tuple

import httpx
from parsel import Selector
from loguru import logger as log

from urls import UrlFilter
from robots import RobotsTxtHandler
from callbacks import callbacks


class Crawler:
    async def __aenter__(self):
        self.session = await httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=5),
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "accept-language": "en-US;en;q=0.9",
                "accept-encoding": "gzip, deflate, br",
            },
        ).__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.session.__aexit__(*args, **kwargs)

    def __init__(
        self, filter: UrlFilter, callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        self.rules = {}  # rules parsed from robots.txt file
        self.rule_bot: RobotsTxtHandler = None  # used during scraping
        self.url_filter: UrlFilter = filter  # url filter class
        self.callbacks = callbacks or {}  # callbacks dict

    def _new_rule_bot(self, base_url: str) -> None:
        self.rule_bot = RobotsTxtHandler(base_url)

    def find_urls(self, responses: List[httpx.Response]) -> List[str]:
        """find valid urls in responses"""
        all_unique_urls = set()
        for response in responses:
            sel = Selector(text=response.text, base_url=str(response.url))
            _urls_in_response = set(
                urljoin(str(response.url), url.strip())
                for url in sel.xpath("//a/@href").getall()
            )
            all_unique_urls |= _urls_in_response

        urls_to_follow = self.url_filter.filter(all_unique_urls)
        log.info(
            f"found {len(urls_to_follow)} urls to follow (from total {len(all_unique_urls)})"
        )
        return urls_to_follow

    def get_site_rules(self) -> dict:
        self.rule_bot.fetch()
        return self.rule_bot.get_rules()

    async def scrape_url(self, url) -> httpx.Response:
        return await self.session.get(url, follow_redirects=True)

    async def scrape(
        self, urls: List[str]
    ) -> Tuple[List[httpx.Response], List[Exception]]:
        """scrape urls and return their responses"""
        responses = []
        failures = []
        log.info(f"scraping {len(urls)} urls")

        tasks = [self.scrape_url(url) for url in urls]
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, httpx.Response):
                responses.append(result)
            else:
                failures.append(result)
        return responses, failures

    async def run(self, start_urls: List[str], max_depth=10) -> None:
        """crawl target to maximum depth or until no more urls are found"""
        url_pool = start_urls
        depth = 0
        while url_pool and depth <= max_depth:
            responses, failures = await self.scrape(url_pool)
            log.info(
                f"depth {depth}: scraped {len(responses)} pages and failed {len(failures)}"
            )
            url_pool = self.find_urls(responses)
            await self.callback(responses)
            depth += 1

    async def callback(self, responses: List[httpx.Response]):
        for response in responses:
            for pattern, fn in self.callbacks.items():
                if pattern.match(str(response.url)):
                    log.debug(f"found matching callback for {response.url}")
                    fn(response=response)


async def run_crawler(seed_urls: list[str], domain: str, sub_domain: str) -> None:
    url_filter = UrlFilter(domain=domain, subdomain=sub_domain)
    async with Crawler(url_filter, callbacks=callbacks) as crawler:
        await crawler.run(seed_urls)
