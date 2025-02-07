import re
from urllib.parse import urljoin, urlparse
from typing import Callable, Dict, List, Any

import nltk
import httpx
from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download("punkt")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("averaged_perceptron_tagger")


from data import save_data_to_json, save_data_for_training

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
    return bool(re.match(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text))


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

    # If <article> is not found, try Reddit/Medium-style posts
    if soup.find("div", class_=re.compile(r"post|content|text|article", re.IGNORECASE)):
        content_candidates.append(
            " ".join(
                p.get_text(strip=True)
                for p in soup.find_all(
                    "div", class_=re.compile(r"post|content|text", re.IGNORECASE)
                )
            )
        )

    # If still no content, fall back to <p> tags, filtering out common non-content elements
    if not content_candidates:
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        filtered_paragraphs = [
            p for p in paragraphs if len(p.split()) > 5
        ]  # Avoid very short text like menu items
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
    soup: BeautifulSoup, url: str, keywords: List[str]
) -> tuple[Dict[str, List[str]], list]:
    """Fetches a webpage and extracts names and keyword matches."""
    try:
        response = await fetch_html(url)
        soup = BeautifulSoup(response.text, "html.parser")

        keyword_results = search_keywords(soup, keywords)

        print("\nKeyword Matches:")
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
        print(f"Error fetching page: {e}")


DATA_EXTRACTION: Dict[str, CallbackFunction] = {
    r".*": extract_names,  # Extract names from text
    r".*": extract_metadata,  # Extract metadata (title, description, keywords)
    r"https?://.*": extract_internal_links,  # Extract internal links
    r"https?://.*": extract_external_links,  # Extract external links
    r"https?://.*": extract_social_links,  # Extract social media links
    r"https?://.*": extract_main_content,  # Extract main site content
    r".*\.(pdf|zip|exe|docx|xlsx|mp4)$": extract_file_downloads,  # Extract downloadable files
}

# Example usage
if __name__ == "__main__":
    import asyncio

    url = "https://scrapfly.io"
    keywords = ["developers", "code", "stuff"]

    asyncio.run(analyze_webpage(url, [kw.strip() for kw in keywords]))
