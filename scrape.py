"""
Generic data extraction utilities. Primarily for static pages.
"""

import re
import asyncio
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from typing import Callable, Dict, List, Any

import httpx
from httpx import URL
from bs4 import BeautifulSoup, Tag
from loguru import logger as log

import nltk
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize

from data import get_starting_data

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ExtractorCallback = Callable[
    [BeautifulSoup, URL], Dict[str, Any] | List[Dict[str, Any]]
]

# ---------------------------------------------------------------------------
# NLTK resource helpers
# ---------------------------------------------------------------------------


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name)


ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("chunkers/maxent_ne_chunker", "maxent_ne_chunker")
ensure_nltk_resource("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

NON_CONTENT_TAGS = {
    "script", "style", "nav", "footer", "aside", "noscript",
    "form", "iframe", "svg", "canvas",
}

NON_CONTENT_ROLES = {"navigation", "complementary", "banner", "contentinfo"}

NON_CONTENT_CLASSES = re.compile(
    r"\b(sidebar|nav|footer|header|cookie|advertisement|ad-|social-"
    r"|comment|comments|widget|menu|banner|popup|modal|overlay)\b",
    re.IGNORECASE,
)


def _is_non_content(tag: Tag) -> bool:
    """Return True for tags that should be excluded from content extraction."""
    if tag.name in NON_CONTENT_TAGS:
        return True
    role = tag.get("role", "")
    if role and role.lower() in NON_CONTENT_ROLES:
        return True
    classes = " ".join(tag.get("class", []))
    if classes and NON_CONTENT_CLASSES.search(classes):
        return True
    return False


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PageResult helpers – normalised output shape
# ---------------------------------------------------------------------------

PAGE_RESULT_FIELDS = [
    "url",
    "timestamp",
    "title",
    "content",
    "description",
    "author",
    "metadata",
    "names",
    "keywords",
    "files",
    "posts",
    "content_blocks",
]


def new_page_result(url: URL) -> dict:
    """Create a blank PageResult dict for merging extractor output into."""
    return {
        "url": str(url),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": "",
        "content": "",
        "description": "",
        "author": "",
        "metadata": {},
        "names": [],
        "keywords": {},
        "files": [],
        "posts": [],
        "content_blocks": [],
    }


# ---------------------------------------------------------------------------
# HTTP fetching
# ---------------------------------------------------------------------------


async def fetch_html(url: URL, timeout: float = 30.0) -> httpx.Response:
    """Fetch HTML content with a browser-like User-Agent."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout), follow_redirects=True
    ) as client:
        response = await client.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        response.raise_for_status()
        return response


# ---------------------------------------------------------------------------
# Name extraction  (NER-based)
# ---------------------------------------------------------------------------

_NAME_PATTERN = re.compile(
    r"\b[A-Z][a-z]+(?:[-'][A-Z][a-z]+)? [A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
    r"(?: [A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?\b"
)


def is_likely_name(text: str) -> bool:
    """Check if text resembles a human name via regex heuristics."""
    return bool(_NAME_PATTERN.match(text))


def extract_names(soup: BeautifulSoup, _: URL) -> Dict[str, List[str]]:
    """Extract probable person names via NER (NLTK)."""
    text_content = " ".join(
        tag.get_text(strip=True)
        for tag in soup.find_all(["h1", "h2", "h3", "p"])
        if not _is_non_content(tag)
    )

    words = word_tokenize(text_content)
    pos_tags = pos_tag(words)

    detected: list[str] = []
    for chunk in ne_chunk(pos_tags):
        if isinstance(chunk, nltk.Tree) and chunk.label() == "PERSON":
            detected.append(" ".join(c[0] for c in chunk))

    seen = set()
    unique = []
    for name in detected:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)

    return {"names": unique if unique else []}


# ---------------------------------------------------------------------------
# Forum post extraction
# ---------------------------------------------------------------------------


def extract_posts(soup: BeautifulSoup, url: URL) -> list[Dict[str, str]]:
    """Extract forum posts / comments from a thread page."""
    posts = []
    for post in soup.find_all(
        "div", class_=re.compile(r"post|comment|message", re.IGNORECASE)
    ):
        author_tag = post.find(
            "a", class_=re.compile(r"user|username|author", re.IGNORECASE)
        )
        content_tag = post.find(
            "div", class_=re.compile(r"content|text|body|message", re.IGNORECASE)
        )
        timestamp_tag = post.find(
            "span", class_=re.compile(r"time|date|timestamp", re.IGNORECASE)
        )

        if author_tag and content_tag:
            post_content = content_tag.get_text(strip=True) or "Unable to retrieve post content"
            timestamp = (
                timestamp_tag.get_text(strip=True) if timestamp_tag else "Unknown"
            )
            posts.append(
                {
                    "author": author_tag.get_text(strip=True).lower(),
                    "content": post_content,
                    "timestamp": timestamp,
                    "url": str(url),
                }
            )

    return posts


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------


def extract_links(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Extract all href links from <a> tags."""
    links = [urljoin(str(base_url), a["href"]) for a in soup.find_all("a", href=True)]
    return {"links": links}


def extract_file_downloads(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Filter links for downloadable file types."""
    links = extract_links(soup, base_url)["links"]
    file_extensions = (
        ".pdf", ".zip", ".exe", ".docx", ".xlsx",
        ".mp4", ".tar", ".doc", ".txt", ".csv",
        ".json", ".xml", ".epub", ".mobi",
    )
    pattern = re.compile(
        rf"({'|'.join(re.escape(ext) for ext in file_extensions)})$",
        re.IGNORECASE,
    )
    return {"files": [link for link in links if pattern.search(link)]}


def extract_internal_links(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Extract links pointing to the same domain."""
    parsed_base = urlparse(str(base_url)).netloc
    links = extract_links(soup, base_url)["links"]
    return {
        "internal_links": [
            link for link in links if urlparse(link).netloc == parsed_base
        ]
    }


def extract_external_links(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Extract links pointing to different domains."""
    parsed_base = urlparse(str(base_url)).netloc
    links = extract_links(soup, base_url)["links"]
    return {
        "external_links": [
            link for link in links if urlparse(link).netloc != parsed_base
        ]
    }


def extract_social_links(soup: BeautifulSoup, base_url: URL) -> Dict[str, List[str]]:
    """Extract links to common social-media platforms."""
    social_domains = (
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "instagram.com",
        "youtube.com",
        "tiktok.com",
        "reddit.com",
        "github.com",
        "medium.com",
    )
    links = extract_links(soup, base_url)["links"]
    return {
        "social_links": [
            link for link in links
            if any(domain in link for domain in social_domains)
        ]
    }


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def extract_metadata(soup: BeautifulSoup, url: URL) -> Dict[str, Any]:
    """Extract page metadata including OG / Twitter cards, author, dates."""
    meta = {}

    # -- <title> --
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    meta["title"] = title

    # -- <meta> tags --
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("property") or "").strip().lower()
        content = tag.get("content", "").strip()
        if not name or not content:
            continue

        if name == "description":
            meta["description"] = content
        elif name == "author":
            meta["author"] = content
        elif name in ("pubdate", "publishdate", "article:published_time"):
            meta.setdefault("published_time", content)
        elif name == "keywords":
            meta["keywords_raw"] = content
        elif name.startswith("og:"):
            meta.setdefault("open_graph", {})
            if isinstance(meta["open_graph"], dict):
                meta["open_graph"][name] = content
        elif name.startswith("twitter:"):
            meta.setdefault("twitter_card", {})
            if isinstance(meta["twitter_card"], dict):
                meta["twitter_card"][name] = content

    # -- canonical link --
    canon = soup.find("link", rel="canonical")
    if canon and canon.get("href"):
        meta["canonical_url"] = urljoin(str(url), canon["href"])

    # -- favicon --
    for link_tag in soup.find_all("link", rel=re.compile(r"icon|shortcut icon", re.I)):
        if link_tag.get("href"):
            meta["favicon"] = urljoin(str(url), link_tag["href"])
            break

    return {"metadata": meta}


# ---------------------------------------------------------------------------
# Content extraction  (text-density scoring)
# ---------------------------------------------------------------------------


def extract_main_content(soup: BeautifulSoup, _: URL) -> Dict[str, Any]:
    """Extract the main article / blog-post content using text-density scoring.

    Returns the cleaned body text under the ``content`` key plus a list of
    block texts under ``content_blocks``.
    """
    # ----- 1. identify candidate content region ----------------------------
    article = None
    for candidate in (
        soup.find("article"),
        soup.find("main"),
        soup.find(attrs={"role": "main"}),
    ):
        if candidate is not None:
            article = candidate
            break

    pool = article if article is not None else soup

    # ----- 2. gather text blocks with density scores -----------------------
    blocks: list[dict] = []
    for el in pool.find_all(
        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "td", "th"]
    ):
        if _is_non_content(el):
            continue

        text = clean_text(el.get_text(separator=" ", strip=True))
        if not text or len(text) < 20:
            continue

        words = len(text.split())
        links = len(el.find_all("a"))
        density = words / (1 + links)
        blocks.append({"text": text, "words": words, "density": density})

    if not blocks:
        return {"content": "", "content_blocks": []}

    # ----- 3. sliding window to find densest region ------------------------
    window = min(5, len(blocks))
    best_start = 0
    best_score = 0.0
    for i in range(len(blocks) - window + 1):
        score = sum(b["density"] for b in blocks[i : i + window])
        if score > best_score:
            best_score = score
            best_start = i

    selected = blocks[best_start : best_start + window]

    # ----- 4. assemble output ----------------------------------------------
    full_text = "\n\n".join(b["text"] for b in selected)

    return {
        "content": full_text,
        "content_blocks": [b["text"] for b in selected],
    }


# ---------------------------------------------------------------------------
# Heading / outline extraction
# ---------------------------------------------------------------------------


def extract_headings(soup: BeautifulSoup, _: URL) -> Dict[str, List[Dict[str, str]]]:
    """Extract the heading structure (h1-h6) for an outline / ToC."""
    outline = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(tag.name[1])
        text = clean_text(tag.get_text(strip=True))
        if text:
            outline.append({"level": level, "text": text})
    return {"headings": outline}


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def search_keywords(soup: BeautifulSoup, keywords: List[str]) -> Dict[str, List[str]]:
    """Search for keywords in the page text and return context snippets."""
    text_content = soup.get_text(" ")
    found: Dict[str, List[str]] = {}

    window_size = 30
    for keyword in keywords:
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        snippet = ""
        found[keyword] = []
        for match in pattern.finditer(text_content):
            start = max(0, match.start() - window_size)
            end = min(len(text_content), match.end() + window_size)
            snippet = text_content[start:end].replace("\n", " ")
            found[keyword].append(snippet)

    return found


# ---------------------------------------------------------------------------
# Wiki helpers
# ---------------------------------------------------------------------------


def extract_wiki_images(soup: BeautifulSoup, url: URL) -> Dict[str, List[str]]:
    """Extract relevant images from a Wikipedia article."""
    images = []
    base = str(url)
    for img in soup.find_all("img", src=True):
        img_url = urljoin(base, img["src"])
        if "wikimedia" in img_url or "upload" in img_url:
            images.append(img_url)
    return {"images": images}


def extract_wiki_content(soup: BeautifulSoup, _: URL) -> Dict[str, str]:
    """Extract the main body paragraphs of a Wikipedia article."""
    content_div = soup.find("div", class_="mw-parser-output")
    paragraphs = content_div.find_all("p") if content_div else []
    paras = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    return {"content": "\n".join(paras)}


# ---------------------------------------------------------------------------
# Extraction pipelines
# ---------------------------------------------------------------------------

META_EXTRACTORS: list[ExtractorCallback] = [
    extract_metadata,
    extract_file_downloads,
    extract_internal_links,
    extract_external_links,
    extract_social_links,
]

CONTENT_EXTRACTORS: list[ExtractorCallback] = [
    extract_main_content,
    extract_headings,
    extract_names,
]

POST_EXTRACTORS: list[ExtractorCallback] = [
    extract_posts,
    extract_metadata,
]

WIKI_EXTRACTORS: list[ExtractorCallback] = [
    extract_internal_links,
    extract_wiki_content,
    extract_wiki_images,
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _merge_into(target: dict, fragment: dict) -> None:
    """Merge an extractor's return value into a PageResult dict.

    Lists are concatenated, dicts are shallow-updated, scalars override.
    """
    for key, value in fragment.items():
        if key not in target:
            target[key] = value
            continue

        if isinstance(target[key], list) and isinstance(value, list):
            target[key].extend(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            target[key].update(value)
        else:
            target[key] = value


async def scrape_page(
    url: URL,
    keywords: List[str] | None = None,
    extractors: list[ExtractorCallback] | None = None,
) -> dict:
    """Fetch a URL, run extractors, and return a normalised PageResult dict."""
    result = new_page_result(url)

    try:
        response = await fetch_html(url)
    except httpx.RequestError as e:
        log.error(f"Request failed for {url}: {e}")
        return result
    except httpx.HTTPStatusError as e:
        log.error(f"HTTP error for {url}: {e.response.status_code}")
        return result

    soup = BeautifulSoup(response.text, "html.parser")

    if keywords:
        result["keywords"] = search_keywords(soup, keywords)

    used_extractors = extractors or (META_EXTRACTORS + CONTENT_EXTRACTORS)
    for fn in used_extractors:
        try:
            data = fn(soup, url)
        except Exception as exc:
            log.warning(f"Extractor {fn.__name__} failed: {exc}")
            continue

        if isinstance(data, list):
            result["posts"] = data
        elif isinstance(data, dict):
            _merge_into(result, data)

    # Promote top-level metadata fields for easy CSV access
    meta = result.get("metadata", {})
    if isinstance(meta, dict):
        result["title"] = meta.get("title", "")
        result["description"] = meta.get("description", "")
        result["author"] = meta.get("author", "")

    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    starting_data = get_starting_data()
    url = starting_data["urls"][0]
    keywords = starting_data["keywords"]

    result = asyncio.run(scrape_page(URL(url), keywords))

    print(f"\n{'='*60}")
    print(f"URL:     {result['url']}")
    print(f"Title:   {result['title']}")
    print(f"Author:  {result['author']}")
    print(f"Time:    {result['timestamp']}")
    print(f"\n--- Content ---\n{result['content'][:500]}...")
    print(f"\n--- Metadata ---")
    for k, v in result.get("metadata", {}).items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print(f"\n--- Names ---")
    print(result.get("names", []))
    print(f"\n--- Keywords ---")
    for kw, occ in result.get("keywords", {}).items():
        print(f"  {kw}: {len(occ)} matches")
