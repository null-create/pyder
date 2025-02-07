import asyncio

from urls import generate_url_filters
from data import DataHandler
from crawler import Crawler
from callbacks import DATA_EXTRACTION


async def main():
    workflow = (
        input("Choose workflow (data_collection/author_detection): ").strip().lower()
    )
    output_format = input("Choose output format (json/csv): ").strip().lower()

    # Load trained model for author detection (if applicable)
    model, tokenizer = None, None
    if workflow == "author_detection":
        from analyze import load_trained_model

        model, tokenizer = load_trained_model("model.pkl", "tokenizer.pkl")

    urls = [
        "https://example.com/blog-post",
        "https://medium.com/@author/article-1",
        "https://www.reddit.com/r/topic/comments/abcdef/post_title",
    ]

    # Initialize DataHandler & Crawler
    crawler = Crawler(
        filters=generate_url_filters(urls),
        data_handler=DataHandler(output_format=output_format),
        workflow=workflow,
        callbacks=DATA_EXTRACTION,
        model=model,
        tokenizer=tokenizer,
    )

    await crawler.run(urls)
    await crawler.close()


if __name__ == "__main__":
    asyncio.run(main())
