import os
import asyncio
from collections import defaultdict
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Tuple

import httpx
from httpx import Response
from parsel import Selector
from bs4 import BeautifulSoup
from loguru import logger as log
from w3lib.url import canonicalize_url

from urls import SitemapParser, UrlFilter, generate_url_filters, get_domain
from scrape import (
    WIKI_EXTRACTORS,
    CONTENT_EXTRACTORS,
    META_EXTRACTORS,
    POST_EXTRACTORS,
    ExtractorCallback,
)
from data import DataHandler, SeedData, get_starting_data


class Crawler:
    async def __aenter__(self):
        self.session = await httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml," "application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
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
        keywords: list[str] | None = None,
        callbacks: list[ExtractorCallback] | None = None,
        search_depth: int | None = None,
        export_data: bool = False,
    ) -> None:
        self.url_filters = filters
        self.data = data_handler
        self.keywords = keywords or []
        self.callbacks = callbacks or []
        self.search_depth = search_depth or 10
        self.export = export_data
        self.visited: set[str] = set()

    async def get(self, url: str) -> httpx.Response:
        """Run a GET request on a given URL."""
        return await self.session.get(url, follow_redirects=True, timeout=3.0)

    def process_responses(self, responses: list[Response]) -> None:
        """Run the configured callbacks over each response and store results."""
        if not responses:
            return

        for response in responses:
            soup = BeautifulSoup(response.text, "html.parser")
            result: dict = {}

            for callback_fn in self.callbacks:
                try:
                    data = callback_fn(soup, str(response.url))
                except Exception as exc:
                    log.warning(f"Callback {callback_fn.__name__} failed: {exc}")
                    continue

                if isinstance(data, list):
                    result.setdefault("posts", [])
                    result["posts"].extend(data)
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if key not in result:
                            result[key] = value
                        elif isinstance(result[key], list) and isinstance(value, list):
                            result[key].extend(value)
                        elif isinstance(result[key], dict) and isinstance(value, dict):
                            result[key].update(value)
                        else:
                            result[key] = value

            result.setdefault("url", str(response.url))

            if self.export and result:
                self.data.store(result)

    def find_urls(self, responses: List[httpx.Response]) -> List[str]:
        """Find valid / followable URLs in responses."""
        if not responses:
            return []

        all_unique_urls: set[str] = set()
        for response in responses:
            sel = Selector(text=response.text, base_url=str(response.url))
            found = set(
                urljoin(str(response.url), url.strip())
                for url in sel.xpath("//a/@href").getall()
            )
            all_unique_urls |= found

        by_domain: dict[str, list[str]] = defaultdict(list)
        for url in all_unique_urls:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            domain = get_domain(hostname)
            by_domain[domain].append(url)

        urls_to_follow: list[str] = []
        for domain, domain_urls in by_domain.items():
            host_filter = self._find_host_filter(domain)
            if host_filter is not None:
                urls_to_follow.extend(host_filter.filter(domain_urls))
            else:
                fallback = self._default_filter(domain)
                urls_to_follow.extend(fallback.filter(domain_urls))

        log.info(
            f"[+] found {len(urls_to_follow)} urls to follow "
            f"(from total {len(all_unique_urls)})"
        )
        return urls_to_follow

    def _find_host_filter(self, domain: str) -> Optional[UrlFilter]:
        """Look up a UrlFilter whose registered domain matches."""
        for urlfilter in self.url_filters.values():
            if urlfilter.domain == domain:
                return urlfilter
        return None

    def _default_filter(self, domain: str) -> UrlFilter:
        """Create a restrictive filter for an unknown domain."""
        return UrlFilter(
            hostname=domain,
            domain=domain,
            subdomain="",
            venture=False,
            follow_paths=None,
        )

    async def retrieve_sites(
        self, urls: List[str]
    ) -> Tuple[List[httpx.Response], List[Exception]]:
        """Scrape URLs and return their responses."""
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
        """Crawl to configured depth or until no more URLs are discovered."""
        parser = SitemapParser()
        url_pool = await parser.get_urls(self.session, start_urls)
        url_pool = url_pool or start_urls

        depth = 0
        while url_pool and depth <= self.search_depth:
            clean_pool: list[str] = []
            for url in url_pool:
                canon = canonicalize_url(url)
                if canon not in self.visited:
                    self.visited.add(canon)
                    clean_pool.append(url)
            url_pool = clean_pool

            if not url_pool:
                log.info("no unvisited URLs remain, stopping")
                break

            responses, failures = await self.retrieve_sites(url_pool)
            log.info(
                f"depth {depth}: scraped {len(responses)} pages, "
                f"{len(failures)} failed"
            )
            self.process_responses(responses)
            url_pool = self.find_urls(responses)
            depth += 1

        if self.export:
            self.data.dump()


async def run_crawler(seed_data: SeedData) -> None:
    outfile = (
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "scraped-data")
        if not seed_data.outfile
        else seed_data.outfile
    )

    async with Crawler(
        filters=generate_url_filters(seed_data.urls),
        data_handler=DataHandler(
            output_file_name=outfile,
            output_format="json",
        ),
        keywords=seed_data.keywords,
        callbacks=CONTENT_EXTRACTORS
        + META_EXTRACTORS
        + POST_EXTRACTORS
        + WIKI_EXTRACTORS,
        search_depth=seed_data.search_depth or 10,
        export_data=True,
    ) as crawler:
        await crawler.run(seed_data.urls)


if __name__ == "__main__":
    seed_data = get_starting_data()

    asyncio.run(run_crawler(seed_data))
