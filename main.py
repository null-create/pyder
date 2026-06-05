from __future__ import annotations

import asyncio

from crawlers import Crawler
from urls import generate_url_filters
from data import DataHandler, get_starting_data
from scrape import META_DATA_EXTRACTORS, CONTENT_EXTRACTORS


async def main() -> None:
    seed_data = get_starting_data()

    async with Crawler(
        filters=generate_url_filters(seed_data["urls"]),
        data_handler=DataHandler(
            output_file_name="scraped-data",
            output_format="json",
        ),
        search_depth=seed_data["search-depth"] or 1,
        callbacks=CONTENT_EXTRACTORS + META_DATA_EXTRACTORS,
        keywords=seed_data["keywords"],
    ) as crawler:
        await crawler.run(seed_data["urls"])


if __name__ == "__main__":
    asyncio.run(main())
