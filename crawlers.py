import os
import json
import asyncio
from urllib.parse import urljoin, urlsplit
from typing import Dict, List, Tuple, Any, Optional

import httpx
from httpx import Response
from parsel import Selector
from bs4 import BeautifulSoup
from numpy import ndarray
from loguru import logger as log
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser, Page
from keras.api.preprocessing.sequence import pad_sequences

from model import Model, load_trained_model
from urls import UrlFilter, generate_url_filters
from scrape import (
    DATA_EXTRACTORS,
    CONTENT_EXTRACTORS,
    WIKI_EXTRACTORS,
    ExtractorCallback,
)
from data import DataHandler, get_starting_data, get_model_and_tokenizer_filenames

load_dotenv()

# Crawler modes and timeouts
ANALYSIS = "text analysis"
DISCOVERY = "data collection"
TIMEOUT = 3000000  # microseconds


class TwitterCrawler:
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self._close_browser()

    def __init__(
        self,
        mode: str,
        tweet_urls: list[str],
        data_handler: DataHandler,
        model: Model = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.urls = tweet_urls
        self.data_handler = data_handler
        self.model = model
        self.tokenizer = tokenizer
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        if mode not in [DISCOVERY, ANALYSIS]:
            raise ValueError(
                f"❌ Incorrect mode. Must be one of: {DISCOVERY}, {ANALYSIS}"
            )

        self.mode = mode

    async def _close_browser(self) -> None:
        """Closes the Playwright browser instance."""
        if self.browser:
            await self.browser.close()
            await self.playwright.stop()
            self.browser = None
            self.page = None

    @staticmethod
    def clean_url(raw_url: str) -> str:
        split_url = urlsplit(raw_url)
        return split_url.scheme + "://" + split_url.netloc + split_url.path

    @staticmethod
    def get_urls(page_content: str, base_url: str) -> list[str]:
        """parse urls from static page content"""
        sel = Selector(text=page_content, base_url=base_url)
        urls_in_page_content = set(
            urljoin(url, url.strip()) for url in sel.xpath("//a/@href").getall()
        )
        return list(urls_in_page_content)

    def predict_author(self, text: str) -> ndarray:
        """sends text to the model to predict whether it was written by a specific author"""
        if self.tokenizer:
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=100)
            return self.model.predict(padded_sequence)[0]

        else:
            return self.model.predict([text])[0]

    async def scrape_tweets(self) -> None:
        """Scrapes tweets using Playwright and captures relevant data."""

        _xhr_calls = []

        def intercept_response(response):
            """Captures all XHR requests made by the page."""
            if response.request.resource_type == "xhr":
                _xhr_calls.append(response)

        self.page.on("response", intercept_response)

        log.info(f"attempting to scrape {len(self.urls)} tweets...")

        total_scraped = 0
        for url in self.urls:
            log.info(f"scraping: {url}...")
            await self.page.goto(url)
            await self.page.wait_for_selector("[data-testid='tweet']", timeout=TIMEOUT)

            # Extract tweet background requests:
            tweet_calls = [
                f for f in _xhr_calls if "TweetResultByRestId" or "TweetDetail" in f.url
            ]
            for xhr in tweet_calls:
                data = await xhr.json()
                tweet_data = (
                    data.get("data", {}).get("tweetResult", {}).get("result", {})
                )
                if tweet_data:
                    post = {
                        "author": tweet_data.get("core", {})
                        .get("user_results", {})
                        .get("result", {})
                        .get("legacy", {})
                        .get("screen_name", ""),
                        "post_content": tweet_data.get("legacy", {}).get(
                            "full_text", ""
                        ),
                        "timestamp": tweet_data.get("legacy", {}).get("created_at", ""),
                        "thread_url": url,
                    }
                    if self.mode == DISCOVERY:
                        self.data_handler.export([post])
                        total_scraped += 1
                    elif self.mode == ANALYSIS:
                        log.warn("not implemented yet")

        log.info(f"scraped {total_scraped} tweets")

    async def close(self) -> None:
        await self._close_browser()


async def run_tweet_crawler(workflow: str, tweet_urls: list[str], author: str) -> None:
    model, tokenizer = None, None
    if workflow == ANALYSIS:
        model_file, tokenizer_file = get_model_and_tokenizer_filenames()
        model, tokenizer = load_trained_model(model_file, tokenizer_file)

    async with TwitterCrawler(
        mode=workflow,
        data_handler=DataHandler(
            output_file_name=f"{author}-tweets", output_format="csv"
        ),
        tweet_urls=tweet_urls,
        model=model,
        tokenizer=tokenizer,
    ) as tc:
        await tc.scrape_tweets()


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
        workflow: str,
        author: str = None,
        model: Model = None,
        tokenizer: Optional[Any] = None,
        keywords: list[str] = None,
        callbacks: list[ExtractorCallback] = None,
        search_depth: int = None,
        export_data: bool = False,
    ) -> None:
        self.url_filters = filters  # url filter class
        self.data = data_handler  # data handler class
        self.workflow = workflow  # workflow type
        self.author = author  # name of author to scrape
        self.model = model  # pre-trained model, if applicable
        self.tokenizer = tokenizer  # ml tokenizer
        self.keywords = keywords or []  # list of keywords to search for
        self.callbacks = callbacks or []  # list data scraper callbacks
        self.search_depth = search_depth or 10  # url pool iterations
        self.export = export_data  # flag for saving json data

        if not self.author and self.workflow == ANALYSIS:
            raise ValueError("❌ No author set for detection mode!")

    async def get(self, url: str) -> httpx.Response:
        """attempts to run a GET request on a given URL"""
        return await self.session.get(url, follow_redirects=True, timeout=3.0)

    def predict_author(self, text: str) -> ndarray:
        """sends text to the model to predict whether it was written by a specific author"""
        if self.tokenizer:
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=100)
            return self.model.predict(padded_sequence)[0]

        else:
            return self.model.predict([text])[0]

    def process_responses(self, responses: list[Response]) -> None:
        """Run a list of generic scraper callbacks over the given list of responses"""
        if len(responses) == 0:
            return

        extracted_data = []
        for response in responses:
            for callback_fn in self.callbacks:
                data = callback_fn(
                    BeautifulSoup(response.text, "html.parser"),
                    str(response.url),
                    self.author,
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
    seed_urls: list[str],
    workflow: str,
    author: str,
    outfile: str = None,
    keywords: list[str] = None,
    model: Model = None,
    tokenizer: Optional[Any] = None,
) -> None:
    if workflow == ANALYSIS:
        model_file, tokenizer_file = get_model_and_tokenizer_filenames()
        model, tokenizer = load_trained_model(model_file, tokenizer_file)

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
        workflow=workflow,
        author=author,
        model=model,
        tokenizer=tokenizer,
        keywords=keywords,
        callbacks=WIKI_EXTRACTORS,
    ) as crawler:
        await crawler.run(seed_urls)


if __name__ == "__main__":
    workflow = DISCOVERY

    starting_data = get_starting_data()
    seed_urls = starting_data["urls"]
    author = starting_data["author"]
    keywords = starting_data["keywords"]
    outfile = starting_data["outfile"]

    asyncio.run(run_crawler(seed_urls, workflow, author, keywords))
