import re
import json
from urllib.parse import urljoin, urlparse
from typing import Callable, Dict, List, Any

import nltk
import httpx
from bs4 import BeautifulSoup
from loguru import logger as log
from playwright.sync_api import sync_playwright

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

from data import save_author_data_for_training

# file for custom call backs defined in EXTRACTION RULES used by the
# crawler class to handle various discoveries and scenaries


# Define type alias for extraction function signatures
CallbackFunction = Callable[[BeautifulSoup, str], Dict[str, Any]]


# used for testing
async def fetch_html(url: str) -> httpx.Response:
    """Fetches the HTML content of the given URL using httpx."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response


def is_likely_name(text: str) -> bool:
    """Uses regex heuristics to check if a string resembles a human name."""
    return bool(
        re.match(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)  # two names
        or re.match(r"\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b", text)  # three names
    )


def extract_names(soup: BeautifulSoup, _: str = "") -> Dict[str, List[str]]:
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

    # Apply regex-based heuristics to detect additional names
    regex_names = re.findall(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", text_content)

    # Combine results and remove duplicates
    unique_names = list(
        set(detected_names + [name for name in regex_names if is_likely_name(name)])
    )

    return {"names": unique_names if unique_names else ["No names found."]}


def extract_author_post(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Extracts posts from a BeautifulSoup object for various social media sites.

    :param soup: BeautifulSoup object containing parsed HTML.
    :param base_url: The base URL of the website (to resolve relative links).
    :return: A list of extracted posts.
    """
    posts = []

    # Define extraction rules per social media site
    extraction_rules = {
        "facebook.com": {
            "author": ["author", "username"],
            "post": ["userContent", "post-text"],
        },
        "twitter.com": {
            "author": ["css-1dbjc4n r-18u37iz r-1wbh5a2"],
            "post": ["css-901oao", "tweet-text"],
        },
        "x.com": {
            "author": ["css-1dbjc4n r-18u37iz r-1wbh5a2"],
            "post": ["css-901oao", "tweet-text"],
        },
        "linkedin.com": {
            "author": ["feed-shared-actor"],
            "post": ["feed-shared-update-v2__description-wrapper"],
        },
        "instagram.com": {
            "author": ["_aacl", "_aaco", "_aacu"],
            "post": ["_a9zs", "_a9zr"],
        },
        "youtube.com": {
            "author": ["ytd-channel-name"],
            "post": ["yt-formatted-string", "comment-text"],
        },
        "tiktok.com": {
            "author": ["css-901oao"],
            "post": ["tiktok-1xg78ey-DivCommentText"],
        },
    }

    # Identify which site's rules to use
    domain = base_url.split("//")[-1].split("/")[0]  # Extract domain from URL
    site_rules = next(
        (rules for site, rules in extraction_rules.items() if site in domain), None
    )

    if not site_rules:
        return []  # If site is unsupported, return empty list

    # Extract author elements
    author_elements = soup.find_all(
        class_=lambda x: x and any(cls in x.lower() for cls in site_rules["author"])
    )

    for author_element in author_elements:
        post_element = author_element.find_next(
            class_=lambda x: x and any(cls in x.lower() for cls in site_rules["post"])
        )
        if post_element:
            post_text = post_element.get_text(strip=True)
            post_link = post_element.find("a", href=True)
            full_url = urljoin(base_url, post_link["href"]) if post_link else None

            posts.append({"text": post_text, "url": full_url})

    return posts


def extract_posts_playwright(url: str, max_posts: int = 5) -> list:
    """
    Extracts posts from a given social media URL using Playwright in headless mode.

    :param url: The URL of the page to scrape.
    :param max_posts: Maximum number of posts to extract.
    :return: A list of extracted posts.
    """
    posts = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Headless mode
        page = browser.new_page()
        page.goto(url, timeout=60000)  # Load page with a timeout

        # Simulate scrolling to load dynamic content
        for _ in range(3):  # Scroll multiple times
            page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

        # Define social media-specific selectors
        selectors = {
            "twitter.com": "article div[lang]",  # Tweet text
            "facebook.com": "div[data-ad-preview]",  # Facebook post
            "linkedin.com": "div.feed-shared-update-v2__description-wrapper",
            "instagram.com": "div._a9zs",
            "youtube.com": "yt-formatted-string#content-text",
            "tiktok.com": "div.tiktok-1xg78ey-DivCommentText",
        }

        domain = url.split("//")[-1].split("/")[0]  # Extract domain
        selector = next((selectors[site] for site in selectors if site in domain), None)

        if selector:
            elements = page.query_selector_all(selector)
            for element in elements[:max_posts]:
                posts.append(element.inner_text().strip())

        browser.close()

    return posts


def extract_links(soup: BeautifulSoup, base_url: str) -> Dict[str, List[str]]:
    """Extracts all links from the webpage."""
    links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]
    return {"links": links}


def extract_named_mentions(soup: BeautifulSoup, _: str) -> Dict[str, List[str]]:
    """Finds mentions of the author's name in text."""
    author_name = ""
    text_content = soup.get_text(" ")
    mentions = re.findall(rf"\b{re.escape(author_name)}\b", text_content, re.IGNORECASE)
    return {"author_mentions": mentions if mentions else ["No mentions found."]}


def extract_file_downloads(soup: BeautifulSoup, base_url: str) -> Dict[str, List[str]]:
    """Filters links for common downloadable file types."""
    links = extract_links(soup, base_url)["links"]
    file_extensions = (".pdf", ".zip", ".exe", ".docx", ".xlsx", ".mp4")
    file_downloads = [
        link
        for link in links
        if re.search(rf"({'|'.join(file_extensions)})$", link, re.IGNORECASE)
    ]
    return {"file_downloads": file_downloads}


def extract_internal_links(soup: BeautifulSoup, base_url: str) -> Dict[str, List[str]]:
    """Extracts internal links that belong to the same domain."""
    parsed_base = urlparse(base_url).netloc
    links = extract_links(soup, base_url)["links"]
    internal_links = [link for link in links if urlparse(link).netloc == parsed_base]
    return {"internal_links": internal_links}


def extract_external_links(soup: BeautifulSoup, base_url: str) -> Dict[str, List[str]]:
    """Extracts external links that belong to different domains."""
    parsed_base = urlparse(base_url).netloc
    links = extract_links(soup, base_url)["links"]
    external_links = [link for link in links if urlparse(link).netloc != parsed_base]
    return {"external_links": external_links}


def extract_metadata(soup: BeautifulSoup, _: str = "") -> Dict[str, str]:
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


def extract_social_links(soup: BeautifulSoup, base_url: str) -> Dict[str, List[str]]:
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


def extract_main_content(soup: BeautifulSoup, _: str = "") -> Dict[str, str]:
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
    filtered_paragraphs = [p for p in paragraphs if len(p.split()) > 5]
    content_candidates.append(" ".join(filtered_paragraphs))

    # Select the longest candidate as the most likely main content
    main_content = max(content_candidates, key=len, default="No main content found.")

    return {"main_content": main_content}


def search_keywords(soup: BeautifulSoup, keywords: List[str]) -> Dict[str, List[str]]:
    """Searches for specific keywords within the webpage text and returns occurrences."""
    text_content = soup.get_text(" ")  # Get all text with spaces
    found_keywords: Dict[str, List[str]] = {}

    for keyword in keywords:
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        matches = pattern.finditer(text_content)

        found_keywords[keyword] = [
            f"...{text_content[max(0, match.start()-30):min(len(text_content), match.end()+30)]}..."
            for match in matches
        ]

    return found_keywords


# used for testing
async def analyze_webpage(
    url: str, keywords: List[str] = None
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
        for pattern, extraction_fn in DATA_EXTRACTION.items():
            if re.match(pattern, url):
                extracted_data.append(extraction_fn(soup, url))

        return (keyword_results, extracted_data)

    except httpx.HTTPError as e:
        log.error(f"❌ Error fetching page: {e}")


DATA_EXTRACTION: Dict[str, CallbackFunction] = {
    r".*": extract_names,  # Extract names from text
    r".*": extract_metadata,  # Extract metadata (title, description, keywords)
    r".*": extract_main_content,  # Extract main site content
    r"https?://.*": extract_internal_links,  # Extract internal links
    r"https?://.*": extract_external_links,  # Extract external links
    r"https?://.*": extract_social_links,  # Extract social media links
    r"https?://.*": extract_author_post,  # Extract any posts by an author
    r".*\.(pdf|zip|exe|docx|xlsx|mp4)$": extract_file_downloads,  # Extract downloadable files
}

# Example usage
if __name__ == "__main__":
    import asyncio

    url = "https://scrapfly.io"
    keywords = ["developers", "code", "stuff"]

    asyncio.run(analyze_webpage(url, [kw.strip() for kw in keywords]))
