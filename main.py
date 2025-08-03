import asyncio

import questionary
from dotenv import load_dotenv

from model import load_trained_model
from urls import generate_url_filters
from data import DataHandler, get_starting_data
from crawlers import Crawler, ANALYSIS, DISCOVERY
from scrape import META_DATA_EXTRACTORS, WIKI_EXTRACTORS, CONTENT_EXTRACTORS

load_dotenv()


async def main() -> None:
    # Get main workflow from user
    workflow = questionary.select(
        "Select workflow", choices=[DISCOVERY, ANALYSIS], default=DISCOVERY
    ).ask()

    # Load trained model for author detection (if applicable)
    model, tokenizer = None, None
    if workflow == ANALYSIS:
        if workflow == "author_detection":
            model, tokenizer = load_trained_model("model.pkl", "tokenizer.pkl")

    # Get starting seed data (urls, keywords, author, etc)
    seed_data = get_starting_data()

    # Initialize DataHandler & Crawler
    async with Crawler(
        filters=generate_url_filters(seed_data["urls"]),
        data_handler=DataHandler(
            output_file_name="scraped-data",
            output_format="csv" if workflow == DISCOVERY else "json",
        ),
        workflow=workflow,
        model=model,
        tokenizer=tokenizer,
        callbacks=META_DATA_EXTRACTORS,
        keywords=seed_data["keywords"],
    ) as crawler:
        await crawler.run(seed_data["urls"])


if __name__ == "__main__":
    asyncio.run(main())
