import asyncio

from dotenv import load_dotenv

from urls import generate_url_filters
from data import DataHandler, get_keywords
from crawler import Crawler, DETECTION, DISCOVERY, get_seed_urls
from extractors import DATA_EXTRACTION

load_dotenv()


async def main():
    # Get main workflow from user
    workflow = (
        input("Choose workflow (data_collection/author_detection): ").strip().lower()
    )

    # Load trained model for author detection (if applicable)
    model, tokenizer = None, None
    if workflow == DETECTION:
        if workflow == "author_detection":
            from model import load_trained_model

            model, tokenizer = load_trained_model("model.pkl", "tokenizer.pkl")

    seed_urls = get_seed_urls()

    # Initialize DataHandler & Crawler
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
        keywords=get_keywords(),
    ) as crawler:
        await crawler.run(seed_urls)


if __name__ == "__main__":
    asyncio.run(main())
