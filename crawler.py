import asyncio
from urllib.parse import urljoin, urlparse
from typing import Callable, Dict, List, Optional, Tuple

import httpx
from parsel import Selector
from loguru import logger as log
from bs4 import BeautifulSoup

from callbacks import CALLBACKS
from data import save_data_to_json
from urls import UrlFilter, get_seed_urls, get_domain, get_subdomain


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
        self,
        filter: UrlFilter,
        callbacks: Optional[Dict[str, Callable]] = None,
        search_depth: int = None,
    ) -> None:
        self.url_filter = filter  # url filter class
        self.search_depth = search_depth or 10  # search depth for each page
        self.callbacks = callbacks or {}  # callbacks dict

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
            f"[+] found {len(urls_to_follow)} urls to follow (from total {len(all_unique_urls)})"
        )
        return urls_to_follow

    async def scrape_url(self, url: str) -> httpx.Response:
        return await self.session.get(url, follow_redirects=True)

    async def scrape(
        self, urls: List[str]
    ) -> Tuple[List[httpx.Response], List[Exception]]:
        """scrape urls and return their responses"""
        responses = []
        failures = []
        log.info(f"[!] scraping {len(urls)} urls")
        tasks = [self.scrape_url(url) for url in urls]
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, httpx.Response):
                responses.append(result)
            else:
                failures.append(result)
        return responses, failures

    async def run(self, start_urls: List[str]) -> None:
        """crawl target to maximum depth or until no more urls are found"""
        url_pool = start_urls
        depth = 0
        while url_pool and depth <= self.search_depth:
            responses, failures = await self.scrape(url_pool)  # scrape url pool
            log.info(
                f"[!] depth {depth}: scraped {len(responses)} pages and failed {len(failures)}"
            )
            url_pool = self.find_urls(responses)  # find next urls to scrape
            await self.callback(responses)  # apply callbacks to the responses
            depth += 1

    async def callback(self, responses: List[httpx.Response]):
        """apply callback function to matching response URLs"""
        for response in responses:
            for pattern, fn in self.callbacks.items():
                if pattern.match(str(response.url)):  # matches a url to a callback
                    log.debug(f"[+] found matching callback for {response.url}")
                    data = fn(
                        soup=BeautifulSoup(response.text, "html.parser"),
                        base_url=response.url,
                    )
                    save_data_to_json(
                        data,
                        get_domain(response.url),
                        f"{get_domain(response.url)}-scrape.json",
                    )


async def run_crawler(seed_urls: list[str], domain: str, sub_domain: str) -> None:
    async with Crawler(
        filter=UrlFilter(domain=domain, subdomain=sub_domain),
        callbacks=CALLBACKS,
    ) as crawler:
        await crawler.run(seed_urls)


if __name__ == "__main__":
    seed_urls = get_seed_urls()
    for url in seed_urls:
        asyncio.run(run_crawler([url], get_domain(url), get_subdomain(url)))
