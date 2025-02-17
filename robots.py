import re
from urllib.parse import urljoin

import httpx
from loguru import logger as log


class Navigator:
    def __init__(self, base_url: str, timeout: int = 3) -> None:
        """
        Initializes the Navigator class with the base URL of the website.

        The Navigator searches for a robots.txt file and/or a sitemap.xml file
        to help the Crawler determine which paths within the site to try and explore.

        :param base_url: The website's base URL (e.g., "https://example.com/").
        :param timeout: Request timeout in seconds (default: 3).
        """
        self.base_url = base_url.rstrip("/") + "/"  # Ensure trailing slash
        self.robots_url = urljoin(self.base_url, "robots.txt")
        self.sitemap_url = urljoin(self.base_url, "sitemap.xml")
        self.discovered_urls = None
        self.timeout = timeout
        self.rules = {}

    async def __aenter__(self):
        self.client = await httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=True
        ).__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.client.__aexit__(*args, **kwargs)

    async def fetch(self) -> None:
        """Fetches the robots.txt file and sitemap.xml file from the website and store them in memory."""
        try:
            response = await self.client.get(self.robots_url)
            if response.status_code == 200:
                self._parse_robots_file(response.text)
            else:
                log.warning(f"❌ failed to retrieve robots.txt: {response.status_code}")
        except httpx.RequestError as e:
            log.error(f"❌ request error: {e}")

        except Exception as e:
            log.error(f"❌ exception occurred: {e}")

    def _parse_robots_file(self, text: str) -> None:
        """Parses the robots.txt file and stores the rules in a dictionary. Clears the internal ruleset each time it is called!"""
        if not text or text == "":
            return

        user_agent = None
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip comments and empty lines

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            key, value = parts[0].strip().lower(), parts[1].strip()
            if key == "user-agent":
                user_agent = value
                if user_agent not in self.rules:
                    self.rules[user_agent] = {
                        "allow": [],
                        "disallow": [],
                        "sitemap": "",
                    }
            elif key in ("disallow", "allow") and user_agent:
                self.rules[user_agent][key].append(value)
            elif key == "sitemap":
                self.rules[key] = value

    async def get_sitemap(self) -> list:
        try:
            response = await self.client.get(self.sitemap_url)
            if response.status_code == 200:
                self._parse_sitemap(response.text)
            else:
                log.warn(f"❌ failed to retrieve site map:\n{response.text}")
        except httpx.RequestError as e:
            log.error(f"❌ request error: {e}")
        except Exception as e:
            log.error(f"❌ exception occurred: {e}")

    def _parse_sitemap(self, data: str) -> None:
        """Parse a site map for urls to follow within a website"""
        if not data:
            return
        urls = []
        url_pattern = re.compile(
            r"<url>.*?<loc>(.*?)</loc>.*?(<lastmod>(.*?)</lastmod>)?.*?(<changefreq>(.*?)</changefreq>)?.*?(<priority>(.*?)</priority>)?.*?</url>",
            re.DOTALL,
        )
        matches = url_pattern.findall(data)
        for match in matches:
            urls.append(
                {
                    "loc": match[0],
                    "lastmod": match[2] if match[2] else None,
                    "changefreq": match[4] if match[4] else None,
                    "priority": float(match[6]) if match[6] else None,
                }
            )

        self.discovered_urls = urls

    def is_allowed(self, user_agent: str, path: str) -> bool:
        """Checks if a given path is allowed for the specified user-agent."""
        user_agent = user_agent.lower()

        # Check specific user-agent rules first, then default '*' rules
        for ua in (user_agent, "*"):
            if ua in self.rules:
                disallowed = any(
                    path.startswith(rule) for rule in self.rules[ua]["disallow"]
                )
                allowed = any(path.startswith(rule) for rule in self.rules[ua]["allow"])
                return allowed or not disallowed

        return True  # Default to allowed if no rules are specified

    def get_rules(self, user_agent: str) -> dict:
        """Returns the allow/disallow rules for a given user-agent."""
        return self.rules.get(
            user_agent.lower(), self.rules.get("*", {"allow": [], "disallow": []})
        )

    def get_sitemap_urls(self) -> list:
        """Returns the list of urls found when parsing the sitemap.xml file"""
        if not self.discovered_urls:
            return []
        urls = [
            self.discovered_urls["loc"] for self.discovered_urls in self.discovered_urls
        ]
        return urls

    def close(self):
        """Closes the HTTP client session."""
        self.client.close()


# Example usage:
if __name__ == "__main__":
    robots_client = Navigator("http://localhost:8000")
    robots_client.fetch()
    print("Rules for '*':", robots_client.get_rules("*"))
    print("Site urls from sitemap:", ", ".join(robots_client.get_sitemap_urls()))
    print("Can we crawl '/private'?:", robots_client.is_allowed("mybot", "/private"))
    robots_client.close()
