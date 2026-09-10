# Pyder

Web scraping and content extraction toolkit for Python.

The default `CMD` runs `crawler.py`. Override it with `docker compose run --rm crawler python <script>` to run `scrape.py` or other commands. Output files written to `/app/data/` persist on the host under `./data/`.

## Modules

- **`scrape.py`** — page-level extraction: fetches HTML, runs extractor pipelines, returns normalised `PageResult` dicts. Handles articles, forum posts, Wikipedia pages, metadata, link analysis, keyword search, and NER-based name extraction.

- **`crawler.py`** — multi-page crawler: discovers URLs via sitemaps, follows internal links to a configurable depth, runs extractors on each page, and exports results.

- **`data.py`** — data handler: accumulates `PageResult` dicts and exports as JSON or CSV, with optional gzip compression.

- **`urls.py`** — sitemap parsing, URL filtering by domain/subdomain/path, and crawl‑boundary enforcement.

## Quick start

```bash
pip install -r requirements.txt
```

Create and configure a JSON file `seed-data.json` with target URLs and optional keywords/search-depth/outfile:

```json
{
    "home": "",
    "urls": [],
    "keywords": [],
    "search-depth": 0,
    "outfile": ""
}
```

Then run:

```bash
python scrape.py          # single-page extraction
python crawler.py         # multi-page crawl
```

## Run With Docker

```bash
# Build and run the crawler
docker compose up --build

# Run interactively (overrides the default CMD)
docker compose run --rm crawler python scrape.py

# Data written to ./data/ is mounted into the container
```

### Programmatic usage

```python
import asyncio
from httpx import URL
from scrape import scrape_page
from data import DataHandler

result = asyncio.run(scrape_page(URL("https://example.com/article")))

handler = DataHandler(output_format="json", output_file_name="output")
handler.store(result)
handler.dump()
```

## Extractor pipelines

| Pipeline | Extractors |
|---|---|
| `META_EXTRACTORS` | metadata (title, OG, Twitter, author, dates, canonical, favicon), file download links, internal/external/social links |
| `CONTENT_EXTRACTORS` | main content (text-density scoring), heading outline, NER names |
| `POST_EXTRACTORS` | forum posts, metadata |
| `WIKI_EXTRACTORS` | internal links, wiki body text, wiki images |

Pass a custom pipeline to `scrape_page(url, extractors=...)` or extend the lists.

## DataHandler output

Both JSON and CSV store the same data:

- **JSON** — a JSON array of `PageResult` dicts (rich, hierarchical)
- **CSV** — flat rows with columns auto-detected from dict keys; nested values are JSON-encoded

## PageResult schema

| Field | Type | Source |
|---|---|---|
| `url` | str | input URL |
| `timestamp` | str (ISO 8601) | crawl time |
| `title` | str | `<title>` / OG |
| `content` | str | `extract_main_content` (cleaned, scored) |
| `description` | str | meta / OG |
| `author` | str | meta author |
| `metadata` | dict | OG, Twitter, canonical, favicon, dates |
| `names` | list[str] | NLTK NER |
| `keywords` | dict | `search_keywords` results |
| `internal_links` | list[str] | same-domain links |
| `external_links` | list[str] | cross-domain links |
| `social_links` | list[str] | Facebook, Twitter, etc. |
| `files` | list[str] | PDF, ZIP, DOCX, etc. |
| `headings` | list[dict] | h1–h6 outline |
| `content_blocks` | list[str] | individual text blocks |
| `posts` | list[dict] | forum posts (if using POST_EXTRACTORS) |
