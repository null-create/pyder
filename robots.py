import httpx
from urllib.parse import urljoin


class RobotsTxtHandler:
    def __init__(self, base_url: str, timeout: int = 5):
        """
        Initializes the RobotsTxtClient with the base URL of the website.
        :param base_url: The website's base URL (e.g., "https://example.com/").
        :param timeout: Request timeout in seconds (default: 5).
        """
        self.base_url = base_url.rstrip("/") + "/"  # Ensure trailing slash
        self.robots_url = urljoin(self.base_url, "robots.txt")
        self.timeout = timeout
        self.content = None
        self.rules = {}
        self.client = httpx.Client(timeout=self.timeout, follow_redirects=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def fetch(self) -> None:
        """Fetches the robots.txt file from the website and stores it in memory."""
        try:
            response = self.client.get(self.robots_url)
            if response.status_code == 200:
                self.content = response.text
                self._parse()
            else:
                print(f"Failed to retrieve robots.txt: {response.status_code}")
        except httpx.RequestError as e:
            print(f"Request error: {e}")

    def _parse(self) -> None:
        """Parses the robots.txt file and stores the rules in a dictionary."""
        self.rules.clear()
        if not self.content:
            return

        user_agent = None
        for line in self.content.splitlines():
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
                    self.rules[user_agent] = {"allow": [], "disallow": []}
            elif key in ("disallow", "allow") and user_agent:
                self.rules[user_agent][key].append(value)

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

    def close(self):
        """Closes the HTTP client session."""
        self.client.close()


# Example usage:
if __name__ == "__main__":
    robots_client = RobotsTxtHandler("https://example.com")
    robots_client.fetch()
    print("Rules for '*':", robots_client.get_rules("*"))
    print("Can we crawl '/private'?:", robots_client.is_allowed("mybot", "/private"))
    robots_client.close()
