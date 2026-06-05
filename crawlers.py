import os
import asyncio
from urllib.parse import urljoin
from typing import Dict, List, Tuple

import httpx
from httpx import Response
from parsel import Selector
from bs4 import BeautifulSoup
from loguru import logger as log

from urls import UrlFilter, generate_url_filters
from scrape import (
    WIKI_EXTRACTORS,
    ExtractorCallback,
)
from data import DataHandler, get_starting_data

### Configurations ###
TIMEOUT = 3000000  # microseconds


class Crawler:
    async def __aenter__(self):
        self.session = await httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "accept-language": "en-US;en;q=0.9",
            },
            follow_redirects=True,
        ).__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.session.__aexit__(*args, **kwargs)

    def __init__(
        self,
        filters: Dict[str, UrlFilter],
        data_handler: DataHandler,
        keywords: list[str] = None,
        callbacks: list[ExtractorCallback] = None,
        search_depth: int = None,
        export_data: bool = False,
    ) -> None:
        self.url_filters = filters  # url filter class
        self.data = data_handler  # data handler class
        self.keywords = keywords or []  # list of keywords to search for
        self.callbacks = callbacks or []  # list data scraper callbacks
        self.search_depth = search_depth or 10  # url pool iterations
        self.export = export_data  # flag for saving json data

    async def get(self, url: str) -> httpx.Response:
        """attempts to run a GET request on a given URL"""
        return await self.session.get(url, follow_redirects=True, timeout=3.0)

    def process_responses(self, responses: list[Response]) -> None:
        """Run a list of generic scraper callbacks over the given list of responses"""
        if len(responses) == 0:
            return

        extracted_data = []
        for response in responses:
            for callback_fn in self.callbacks:
                data = callback_fn(
                    BeautifulSoup(response.text, "html.parser"), str(response.url)
                )
                if isinstance(data, list):
                    extracted_data += data
                elif isinstance(data, dict):
                    extracted_data.append(data)
                else:
                    log.warning(f"Unexpected returned data type: {type(data)}")

        if self.export:
            self.data.export(extracted_data)

    def find_urls(self, responses: List[httpx.Response]) -> List[str]:
        """find valid urls in responses"""
        if len(responses) == 0:
            return []

        all_unique_urls = set()
        urls_to_follow = []
        for response in responses:
            sel = Selector(text=response.text, base_url=str(response.url))
            _urls_in_response = set(
                urljoin(str(response.url), url.strip())
                for url in sel.xpath("//a/@href").getall()
            )
            all_unique_urls |= _urls_in_response

        if response.url.host in self.url_filters:
            urls_to_follow += self.url_filters[response.url.host].filter(
                all_unique_urls
            )
        else:
            urls_to_follow += list(all_unique_urls)

        log.info(
            f"[+] found {len(urls_to_follow)} urls to follow (from total {len(all_unique_urls)})"
        )
        return urls_to_follow

    async def scrape(
        self, urls: List[str]
    ) -> Tuple[List[httpx.Response], List[Exception]]:
        """scrape urls and return their responses"""
        responses = []
        failures = []
        log.info(f"🔎 scraping {len(urls)} urls")
        tasks = [self.get(url) for url in urls]
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
            responses, failures = await self.scrape(url_pool)
            log.info(
                f"depth {depth}: scraped {len(responses)} pages and failed {len(failures)}"
            )
            self.process_responses(responses)
            url_pool = self.find_urls(responses)
            depth += 1


async def run_crawler(
    seed_urls: list[str], outfile: str = None, keywords: list[str] = None
) -> None:
    outfile = (
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "scraped-data")
        if not outfile
        else outfile
    )

    async with Crawler(
        filters=generate_url_filters(seed_urls),
        data_handler=DataHandler(
            output_file_name=outfile,
            output_format="json",
        ),
        keywords=keywords,
        callbacks=WIKI_EXTRACTORS,
    ) as crawler:
        await crawler.run(seed_urls)


if __name__ == "__main__":
    starting_data = get_starting_data()
    seed_urls = starting_data["urls"]
    keywords = starting_data["keywords"]
    outfile = starting_data["outfile"]

    asyncio.run(run_crawler(seed_urls, outfile, keywords))
