import os
import re
import asyncio
from urllib.parse import urljoin
from typing import Dict, List, Tuple, Any, Optional

import httpx
from httpx import Response, URL
from parsel import Selector
from bs4 import BeautifulSoup
from numpy import ndarray
from loguru import logger as log
from dotenv import load_dotenv

from keras.api.preprocessing.sequence import pad_sequences

from extractors import POST_CONTENT_EXTRACTORS, ExtractorCallback
from model import Model, load_trained_model
from urls import UrlFilter, generate_url_filters
from data import DataHandler, get_starting_data, get_model_and_tokenizer_filenames

load_dotenv()

# Crawler modes
DISCOVERY = "data_collection"
DETECTION = "author_detection"


class Crawler:
    async def __aenter__(self):
        self.session = await httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=5),
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "accept-language": "en-US;en;q=0.9",
            },
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
        self.author = author or ""  # name of author to scrape
        self.model = model  # pre-trained model, if applicable
        self.tokenizer = tokenizer  # ml tokenizer
        self.keywords = keywords or []  # list of keywords to search for
        self.callbacks = callbacks or []  # data extractor callbacks
        self.search_depth = search_depth or 10  # url pool iterations
        self.export = export_data  # flag for saving json data

        if self.author == "" and self.workflow == DETECTION:
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

    def run_callbacks(self, responses: list[Response]) -> None:
        """Run a list of data extraction callbacks over the given list of responses"""
        if len(responses) == 0:
            return

        extracted_data = []
        for response in responses:
            for callback in self.callbacks:
                data = callback(
                    BeautifulSoup(response.text, "html.parser"),
                    response.url,
                    self.author,
                )
                if isinstance(data, list):
                    extracted_data += data
                elif isinstance(data, dict):
                    extracted_data.append(data)

        if self.export:
            self.data.export(extracted_data)

    def extract_posts(self, soup: BeautifulSoup, url: URL) -> list[Dict[str, Any]]:
        posts = []
        for post in soup.find_all(
            "div", class_=re.compile(r"post|comment|message", re.IGNORECASE)
        ):
            author_tag = post.find(
                "a", class_=re.compile(r"user|username|author", re.IGNORECASE)
            )
            content_tag = post.find(
                "div", class_=re.compile(r"content|text|body", re.IGNORECASE)
            )
            timestamp_tag = post.find(
                "span", class_=re.compile(r"time|date", re.IGNORECASE)
            )

            if author_tag and content_tag:
                post_author = author_tag.get_text(strip=True).lower()
                if post_author == self.author:
                    post_content = (
                        content_tag.get_text(strip=True)
                        if post_content
                        else "Unable to retrive post content"
                    )
                    timestamp = (
                        timestamp_tag.get_text(strip=True)
                        if timestamp_tag
                        else "Unknown"
                    )
                    posts.append(
                        {
                            "author": post_author,
                            "post_content": post_content,
                            "timestamp": timestamp,
                            "thread_url": str(url),
                        }
                    )

        return posts

    def process_responses(self, responses: list[Response]) -> None:
        """
        Processes the responses returned from the initial scrape.
        Exctacts posts content, and will attempt to run a preditiction using an
        instantiated model to try and guess if this post was written by our author
        if the crawler is in detection mode
        """
        if len(responses) == 0:
            return

        extracted_data = []
        for response in responses:
            posts = self.extract_posts(
                BeautifulSoup(response.text, "html.parser"), response.url
            )

            if self.workflow == DETECTION and self.model:
                for post in posts:
                    guess = self.predict_author(post["post_content"])
                    post["author_prediction"] = guess
                    log.info(f"🕵️‍♀️ {post['author']} = {self.author} likelyhood: {guess}")

            extracted_data += posts

        if self.export:
            self.data.export(extracted_data)

    def find_urls(self, responses: List[httpx.Response]) -> List[str]:
        """find valid urls in responses"""
        if len(responses) == 0:
            return []

        all_unique_urls = set()
        for response in responses:
            sel = Selector(text=response.text, base_url=str(response.url))
            _urls_in_response = set(
                urljoin(str(response.url), url.strip())
                for url in sel.xpath("//a/@href").getall()
            )
            all_unique_urls |= _urls_in_response

        if response.url.host in self.url_filters:
            urls_to_follow = self.url_filters[response.url.host].filter(all_unique_urls)
        else:
            urls_to_follow = list(all_unique_urls)

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
    keywords: list[str] = None,
    model: Model = None,
    tokenizer: Optional[Any] = None,
) -> None:
    if workflow == DETECTION:
        model_file, tokenizer_file = get_model_and_tokenizer_filenames()
        model, tokenizer = load_trained_model(model_file, tokenizer_file)

    async with Crawler(
        filters=generate_url_filters(seed_urls),
        data_handler=DataHandler(
            output_file_name=os.path.join("data", "scraped-data"),
            output_format="csv" if workflow == DISCOVERY else "json",
        ),
        workflow=workflow,
        author=author,
        model=model,
        tokenizer=tokenizer,
        keywords=keywords,
        callbacks=POST_CONTENT_EXTRACTORS,
        export_data=True,
    ) as crawler:
        await crawler.run(seed_urls)


if __name__ == "__main__":
    workflow = DISCOVERY
    starting_data = get_starting_data()
    author = starting_data["author"]
    keywords = starting_data["keywords"]
    seed_urls = starting_data["urls"]

    asyncio.run(run_crawler(seed_urls, workflow, author, keywords))
