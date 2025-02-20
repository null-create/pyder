import posixpath
from typing import List, Pattern, Dict
from urllib.parse import urlparse

from tldextract import tldextract
from w3lib.url import canonicalize_url
from loguru import logger as log


class UrlFilter:
    def __init__(
        self,
        hostname: str,
        domain: str = None,
        subdomain: str = None,
        venture: bool = False,
        follow_paths: List[Pattern] = None,
    ) -> None:
        self.hostname = hostname  # site host
        self.domain = domain or ""  # restrict filtering to specific TLD
        self.subdomain = subdomain or ""  # restrict filtering to specific subdomain
        self.venture = venture  # whether to allow off-site urls
        self.follow_paths = follow_paths or []  # list of regex path patterns
        self.seen = set()  # visited URLs

        log.info(
            f"filter created for domain {self.subdomain}.{self.domain} with follow rules {follow_paths}"
        )

    def is_valid_ext(self, url: str) -> bool:
        """ignore non-crawlable documents"""
        return (
            posixpath.splitext(urlparse(url).path)[1].lower() not in IGNORED_EXTENSIONS
        )

    def is_valid_scheme(self, url: str) -> bool:
        """ignore non http/s links"""
        return urlparse(url).scheme in ["https", "http"]

    def is_valid_domain(self, url: str) -> bool:
        """ignore offsite urls"""
        parsed = tldextract.extract(url)
        return (
            parsed.registered_domain == self.domain
            and parsed.subdomain == self.subdomain
        )

    def is_related(self, url: str) -> bool:
        domain = get_domain(url)
        return self.domain == domain

    def is_valid_path(self, url: str) -> bool:
        """ignore urls of undesired paths"""
        if not self.follow_paths:
            return True
        path = urlparse(url).path
        for pattern in self.follow_paths:
            if pattern.match(path):
                return True
        return False

    def is_new(self, url: str) -> bool:
        """ignore visited urls (in canonical form)"""
        return canonicalize_url(url) not in self.seen

    def filter(self, urls: List[str]) -> List[str]:
        """filter list of urls"""
        found = []
        for url in urls:
            if not self.is_valid_scheme(url):
                log.debug(f"drop ignored scheme {url}")
                continue
            if not self.venture and not self.is_valid_domain(url):
                log.debug(f"drop offsite url {url}")
                continue
            if not self.is_valid_ext(url):
                log.debug(f"drop ignored extension {url}")
                continue
            if not self.is_valid_path(url):
                log.debug(f"drop ignored path {url}")
                continue
            if not self.is_new(url):
                log.debug(f"drop duplicate {url}")
                continue
            self.seen.add(canonicalize_url(url))
            found.append(url)
        return found


def generate_url_filters(urls: List[str], venture: bool = True) -> Dict[str, UrlFilter]:
    """create a dictionary of URL filters based off the given set of URLs"""
    url_filters: Dict[str, UrlFilter] = {}

    for url in urls:
        parsed_url = urlparse(url)
        domain = get_domain(parsed_url.netloc)
        subdomain = get_subdomain(parsed_url.netloc)

        if parsed_url.hostname and parsed_url.hostname not in url_filters:
            url_filters[parsed_url.hostname] = UrlFilter(
                hostname=parsed_url.hostname,
                domain=domain,
                subdomain=subdomain,
                venture=venture,
            )

    return url_filters


def get_domain(url: str) -> str:
    extracted = tldextract.extract(url)
    return extracted.domain if extracted.subdomain else ""


def get_subdomain(url: str) -> str:
    extracted = tldextract.extract(url)
    return extracted.subdomain if extracted.subdomain else ""


IGNORED_EXTENSIONS = [
    # archives
    "7z",
    "7zip",
    "bz2",
    "rar",
    "tar",
    "tar.gz",
    "xz",
    "zip",
    # images
    "mng",
    "pct",
    "bmp",
    "gif",
    "jpg",
    "jpeg",
    "png",
    "pst",
    "psp",
    "tif",
    "tiff",
    "ai",
    "drw",
    "dxf",
    "eps",
    "ps",
    "svg",
    "cdr",
    "ico",
    # audio
    "mp3",
    "wma",
    "ogg",
    "wav",
    "ra",
    "aac",
    "mid",
    "au",
    "aiff",
    # video
    "3gp",
    "asf",
    "asx",
    "avi",
    "mov",
    "mp4",
    "mpg",
    "qt",
    "rm",
    "swf",
    "wmv",
    "m4a",
    "m4v",
    "flv",
    "webm",
    # office suites
    "xls",
    "xlsx",
    "ppt",
    "pptx",
    "pps",
    "doc",
    "docx",
    "odt",
    "ods",
    "odg",
    "odp",
    # other
    "css",
    "pdf",
    "exe",
    "bin",
    "rss",
    "dmg",
    "iso",
    "apk",
]
