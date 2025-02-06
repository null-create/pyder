import re
from urllib.parse import urljoin, urlparse
from typing import Callable, Dict, List, Any

import httpx
from bs4 import BeautifulSoup

from data import save_data_to_json

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


def extract_names(soup: BeautifulSoup, _: str = "") -> Dict[str, List[str]]:
    """Extracts names from headings and paragraph tags."""
    return {
        "names": [
            tag.get_text(strip=True) for tag in soup.find_all(["h1", "h2", "h3", "p"])
        ]
    }


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


def extract_metadata(soup: BeautifulSoup, _: str) -> Dict[str, str]:
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


def extract_names(soup: BeautifulSoup, _: str) -> List[str]:
    """Extracts potential names from headings and paragraph text."""
    return [tag.get_text(strip=True) for tag in soup.find_all(["h1", "h2", "h3", "p"])]


def extract_data(html: str, base_url: str, writeout: bool = True) -> Dict[str, Any]:
    """Applies a callback based on url patterns."""
    soup = BeautifulSoup(html, "html.parser")
    extracted_data: Dict[str, Any] = {}

    for pattern, callback in CALLBACKS.items():
        if pattern.match(base_url):
            extracted_data.update(callback(soup, base_url))

    if writeout:
        save_data_to_json(extracted_data, base_url, f"{urlparse(base_url)}.json")

    return extracted_data


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


async def analyze_webpage(url: str, keywords: List[str]) -> None:
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

    except httpx.HTTPError as e:
        print(f"Error fetching page: {e}")


CALLBACKS: Dict[str, CallbackFunction] = {
    r".*": extract_names,  # Extract names from text
    r".*": extract_named_mentions,  # Finds author's name in text
    r".*": extract_metadata,  # Extract metadata (title, description, keywords)
    r"https?://.*": extract_links,  # Extract all links
    r"https?://.*": extract_internal_links,  # Extract internal links
    r"https?://.*": extract_external_links,  # Extract external links
    r"https?://.*": extract_social_links,  # Extract social media links
    r".*\.(pdf|zip|exe|docx|xlsx|mp4)$": extract_file_downloads,  # Extract downloadable files
}

if __name__ == "__main__":
    import asyncio

    url = "https://scrapfly.io"
    keywords = ["developers", "code", "stuff"]

    asyncio.run(analyze_webpage(url, [kw.strip() for kw in keywords]))
