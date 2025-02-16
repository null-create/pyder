import os
import re
import asyncio
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Tuple, Any, Optional

import httpx
from httpx import Response
from parsel import Selector
from loguru import logger as log
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from keras.api.preprocessing.sequence import pad_sequences

from model import Model, load_trained_model
from extractors import DATA_EXTRACTION, ExtractorCallback
from urls import UrlFilter, get_seed_urls, generate_url_filters
from data import DataHandler, get_keywords, get_model_and_tokenizer_filenames

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
        model: Model = None,
        tokenizer: Optional[Any] = None,
        callbacks: list[ExtractorCallback] = None,
        keywords: list[str] = None,
        search_depth: int = None,
        export_data: bool = False,
    ) -> None:
        self.url_filters = filters  # url filter class
        self.data = data_handler  # data handler class
        self.workflow = workflow  # workflow type
        self.model = model  # pre-trained model, if applicable
        self.tokenizer = tokenizer  # ml tokenizer
        self.callbacks = callbacks or {}  # callbacks dict
        self.keywords = keywords or []  # list of keywords to search for
        self.search_depth = search_depth or 10  # url pool iterations
        self.export = export_data  # flag for saving json data

    def predict_author(self, text: str) -> str:
        """sends text to the model to predict whether it was written by a specific author"""
        if self.tokenizer:
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=100)
            prediction = self.model.predict(padded_sequence)[0]
            return "Author" if prediction > 0.5 else "Other"

        else:
            return "Author" if self.model.predict([text])[0] == 1 else "Other"

    def process_responses(self, responses: list[Response]) -> None:
        """processes the responses returned from the initial scrape."""
        extracted_data = {"url": responses[0].url}

        for response in responses:
            for extractor_fn in self.callbacks:
                extracted_data.update(
                    extractor_fn(
                        BeautifulSoup(response.text, "html.parser"),
                        response.url,
                    )
                )

            if self.workflow == DETECTION and self.model:
                log.info("[!] running author prediction...")
                extracted_data["author_prediction"] = self.predict_author(
                    extracted_data.get("main_content", "")
                )
                log.info(f"[+] model guess: {extracted_data['author_prediction']}")

        if self.export:
            self.data.export(extracted_data)

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

        if response.url.host in self.url_filters:
            urls_to_follow = self.url_filters[response.url.host].filter(all_unique_urls)
        else:
            urls_to_follow = all_unique_urls

        log.info(
            f"[+] found {len(urls_to_follow)} urls to follow (from total {len(all_unique_urls)})"
        )
        return urls_to_follow

    async def get(self, url: str) -> httpx.Response:
        return await self.session.get(url, follow_redirects=True, timeout=1.0)

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
                f"[!] depth {depth}: scraped {len(responses)} pages and failed {len(failures)}"
            )
            self.process_responses(responses)
            url_pool = self.find_urls(responses)
            depth += 1


async def run_crawler(
    seed_urls: list[str],
    workflow: str,
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
            output_file_name="scraped-data",
            output_format="csv" if workflow == DISCOVERY else "json",
        ),
        workflow=workflow,
        model=model,
        tokenizer=tokenizer,
        callbacks=DATA_EXTRACTION,
        keywords=keywords,
    ) as crawler:
        await crawler.run(seed_urls)


if __name__ == "__main__":
    workflow = DISCOVERY
    keywords = get_keywords()
    seed_urls = get_seed_urls()

    asyncio.run(run_crawler(seed_urls, workflow, keywords))
