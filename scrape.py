"""
Generic data extraction utilities. Primariliy for static pages
"""

import re
import asyncio
from urllib.parse import urljoin, urlparse
from typing import Callable, Dict, List, Any

import httpx
from httpx import URL
from bs4 import BeautifulSoup
from loguru import logger as log

import nltk
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize

from data import get_starting_data

# Define type alias for extraction function signatures
ExtractorCallback = Callable[
    [BeautifulSoup, URL], Dict[str, Any] | List[Dict[str, Any]]
]


# Add or download resources as necessary
def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name)


# nltk.download("punkt_tab")
# nltk.download("maxent_ne_chunker_tab")
# nltk.download("averaged_perceptron_tagger_eng")

# See: https://www.nltk.org/nltk_data/
ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("chunkers/maxent_ne_chunker", "maxent_ne_chunker")
ensure_nltk_resource("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")


async def fetch_html(url: URL) -> httpx.Response:
    """Fetches the HTML content of the given URL using httpx."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "accept-language": "en-US;en;q=0.9",
            },
        )
        response.raise_for_status()
        return response


def is_likely_name(text: str) -> bool:
    """Uses regex heuristics to check if a string resembles a human name."""
    return bool(
        re.match(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)  # two names
        or re.match(r"\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b", text)  # three names
    )


def extract_names(soup: BeautifulSoup, _: URL) -> Dict[str, List[str]]:
    """Extracts potential names of people from the webpage content using NER and regex."""
    text_content = " ".join(
        tag.get_text(strip=True) for tag in soup.find_all(["h1", "h2", "h3", "p"])
    )

    # Tokenize and apply Named Entity Recognition (NER)
    words = word_tokenize(text_content)
    pos_tags = pos_tag(words)
    named_entities = ne_chunk(pos_tags)

    # Extract named entities recognized as people
    detected_names = []
    for chunk in named_entities:
        if isinstance(chunk, nltk.Tree) and chunk.label() == "PERSON":
            name = " ".join(c[0] for c in chunk)
            detected_names.append(name)

    # Remove duplicates
    unique_names = set()
    unique_names = [
        name
        for name in detected_names
        if not (name in unique_names or unique_names.add(name))
    ]

    return {"names": unique_names if unique_names else ["No names found"]}


def extract_posts(soup: BeautifulSoup, url: URL) -> list[Dict[str, str]]:
    """Extracts forum posts by the target author from a given thread URL."""

    # Forum-specific extraction logic
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
            post_content = (
                content_tag.get_text(strip=True)
                if post_content
                else "Unable to retrive post content"
            )
            timestamp = (
                timestamp_tag.get_text(strip=True) if timestamp_tag else "Unknown"
            )
            posts.append(
                {
                    "author": post_author,
                    "content": post_content,
                    "timestamp": timestamp,
                    "url": str(url),
                }
            )

    return posts


def extract_links(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Extracts all links from the webpage."""
    links = [urljoin(str(base_url), a["href"]) for a in soup.find_all("a", href=True)]
    return {"links": links}


def extract_file_downloads(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Filters links for common downloadable file types."""
    links = extract_links(soup, base_url)["links"]
    file_extensions = (
        ".pdf",
        ".zip",
        ".exe",
        ".docx",
        ".xlsx",
        ".mp4",
        ".tar",
        ".doc",
        ".txt",
    )
    file_downloads = [
        link
        for link in links
        if re.search(rf"({'|'.join(file_extensions)})$", link, re.IGNORECASE)
    ]
    return {"file_downloads": file_downloads}


def extract_internal_links(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Extracts internal links that belong to the same domain."""
    parsed_base = urlparse(str(base_url)).netloc
    links = extract_links(soup, base_url)["links"]
    internal_links = [link for link in links if urlparse(link).netloc == parsed_base]
    return {"internal_links": internal_links}


def extract_external_links(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Extracts external links that belong to different domains."""
    parsed_base = urlparse(str(base_url)).netloc
    links = extract_links(soup, base_url)["links"]
    external_links = [link for link in links if urlparse(link).netloc != parsed_base]
    return {"external_links": external_links}


def extract_metadata(soup: BeautifulSoup, _: URL) -> Dict[str, str]:
    """Extracts metadata such as title, description, and keywords."""
    title = soup.title.string.strip() if soup.title else "No Title"
    description = soup.find("meta", attrs={"name": "description"})
    keywords = soup.find("meta", attrs={"name": "keywords"})

    return {
        "title": title,
        "description": (
            description["content"].strip() if description else "No Description"
        ),
        "keywords": keywords["content"].strip() if keywords else "No Keywords",
    }


def extract_social_links(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Extracts social media links from the webpage."""
    social_domains = (
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "instagram.com",
        "youtube.com",
        "tiktok.com",
    )
    links = extract_links(soup, base_url)["links"]
    social_links = [
        link for link in links if any(domain in link for domain in social_domains)
    ]
    return {"social_links": social_links}


# WIKI specific extractors


def extract_wiki_images(soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
    """Extracts relevant images from a WIKI article."""
    images = []
    for img in soup.find_all("img"):
        img_url = urljoin(base_url, img["src"])  # Convert relative URLs to absolute
        if "wikimedia" in img_url or "upload" in img_url:  # Filter for relevant images
            images.append(img_url)
    return {"images": images}


def extract_wiki_content(soup: BeautifulSoup, _: URL) -> Dict[str, str]:
    """Extracts the main content of the Wiki article (paragraphs only)."""
    content_div = soup.find("div", {"class": "mw-parser-output"})
    paragraphs = content_div.find_all("p") if content_div else []
    content = "\n".join(
        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
    )
    return {"content": content}


# Article/blog-post extractors


def extract_main_content(soup: BeautifulSoup, _: URL) -> Dict[str, str]:
    """Extracts the main content of an article, blog post, or social media post."""

    # Try extracting from standard article-like structures
    content_candidates = []

    # Look for <article> tag first (common in blogs & news sites)
    if soup.find("article"):
        content_candidates.append(
            " ".join(p.get_text(strip=True) for p in soup.find("article").find_all("p"))
        )

    # Look for Reddit/Medium-style posts
    if soup.find("div", class_=re.compile(r"post|content|text|article", re.IGNORECASE)):
        content_candidates.append(
            " ".join(
                p.get_text(strip=True)
                for p in soup.find_all(
                    "div", class_=re.compile(r"post|content|text", re.IGNORECASE)
                )
            )
        )

    # Get all <p> tags, filtering out common non-content elements
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    content_candidates += [p for p in paragraphs if len(p.split()) > 5]

    # Select the longest candidate as the most likely main content
    main_content = max(content_candidates, key=len, default="No main content found.")

    return {"main_content": main_content, "all_content": content_candidates}


def search_keywords(soup: BeautifulSoup, keywords: List[str]) -> Dict[str, List[str]]:
    """Searches for specific keywords within the webpage text and returns occurrences."""
    text_content = soup.get_text(" ")  # Get all text with spaces
    found_keywords: Dict[str, List[str]] = {}

    window_size = 30
    for keyword in keywords:
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        matches = pattern.finditer(text_content)

        found_keywords[keyword] = [
            f"...{text_content[max(0, match.start()-window_size):min(len(text_content), match.end()+window_size)]}...".replace(
                "\n", ""
            )
            for match in matches
        ]

    return found_keywords


### Testing Utilities


# used for testing
async def scrape_page(
    url: URL, keywords: List[str] = None
) -> tuple[Dict[str, List[str]], list]:
    """Fetches a webpage and extracts names and keyword matches."""
    try:
        response = await fetch_html(url)
        soup = BeautifulSoup(response.text, "html.parser")

        if keywords:
            keyword_results = search_keywords(soup, keywords)

            log.info("\nKeyword Matches:")
            for keyword, occurrences in keyword_results.items():
                print(f"\nKeyword: {keyword}")
                if occurrences:
                    print("\n".join(occurrences[:5]))  # Show first 5 matches
                else:
                    print("No occurrences found.")

        extracted_data = []
        for extraction_fn in META_DATA_EXTRACTORS:
            data = extraction_fn(soup, url)
            if isinstance(data, list):
                extracted_data += data
            elif isinstance(data, dict):
                extracted_data.append(data)

        for extraction_fn in CONTENT_EXTRACTORS:
            data = extraction_fn(soup, url)
            if isinstance(data, list):
                extracted_data += data
            elif isinstance(data, dict):
                extracted_data.append(data)

        return (keyword_results, extracted_data)

    except httpx.HTTPError as e:
        log.error(f"❌ Error fetching page: {e}")


# Generic data extractors
META_DATA_EXTRACTORS: list[ExtractorCallback] = [
    extract_metadata,  # Extract metadata (title, description, keywords)
    extract_internal_links,  # Extract internal links
    extract_external_links,  # Extract external links
    extract_social_links,  # Extract social media links
]

# WIKI page extractors
WIKI_EXTRACTORS: list[ExtractorCallback] = [
    extract_internal_links,  # Extract internal links
    extract_wiki_content,  # Extract WIKI article content
    extract_wiki_images,  # Extract WIKI image links
]

# Post and author-specific data extractors
CONTENT_EXTRACTORS: list[ExtractorCallback] = [
    extract_posts,  # extract posts by an author
    extract_main_content,  # extract the main content of a page
    extract_file_downloads,  # Extract downloadable files
]


if __name__ == "__main__":
    starting_data = get_starting_data()
    url = starting_data["urls"][0]
    author = starting_data["author"]
    keywords = starting_data["keywords"]

    asyncio.run(scrape_page(URL(url), keywords))
