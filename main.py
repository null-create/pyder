from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from urls import generate_url_filters
from data import DataHandler, get_starting_data
from crawlers import Crawler
from scrape import DATA_EXTRACTORS, WIKI_EXTRACTORS, CONTENT_EXTRACTORS

load_dotenv()


async def main() -> None:
    # Get starting seed data (urls, keywords, author, etc)
    seed_data = get_starting_data()

    # Initialize DataHandler & Crawler
    async with Crawler(
        filters=generate_url_filters(seed_data["urls"]),
        data_handler=DataHandler(
            output_file_name="scraped-data",
            output_format="json",
        ),
        search_depth=seed_data["search-depth"],
        callbacks=WIKI_EXTRACTORS,
        keywords=seed_data["keywords"],
    ) as crawler:
        await crawler.run(seed_data["urls"])


if __name__ == "__main__":
    asyncio.run(main())
