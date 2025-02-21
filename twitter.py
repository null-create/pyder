import re
import json
import asyncio
from typing import Dict, List, Any, Callable

import twint
from httpx import URL
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

from data import DataHandler

###############################################
#
#   Twitter-specific data extraction stuff
#
###############################################

TIMEOUT = 30000000


def extract_tweet_urls(soup: BeautifulSoup, _: URL, target_author: str) -> List[str]:
    """
    Extracts URLs of all tweets from the author's Twitter home page.

    :param soup: Parsed BeautifulSoup object of the Twitter page.
    :param target_author: The username of the author whose tweets we are extracting.
    :return: List of tweet URLs.
    """
    tweet_urls = []

    # Find all <div> elements with data-testid="tweet"
    for tweet in soup.find_all("div", attrs={"data-testid": "tweet"}):
        link_tag = tweet.find("a", href=re.compile(rf"^/{target_author}/status/\d+$"))
        if link_tag:
            tweet_url = f"https://twitter.com{link_tag['href']}"
            tweet_urls.append(tweet_url)

    return list(set(tweet_urls))  # Remove duplicates


def scrape_tweet(url: str) -> dict:
    """
    Scrape a single tweet page for Tweet thread e.g.:
    https://twitter.com/Scrapfly_dev/status/1667013143904567296
    Return parent tweet, reply tweets and recommended tweets
    """
    _xhr_calls = []

    def intercept_response(response):
        """capture all background requests and save them"""
        # we can extract details from background requests
        if response.request.resource_type == "xhr":
            _xhr_calls.append(response)
        return response

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # enable background request intercepting:
        page.on("response", intercept_response)
        # go to url and wait for the page to load
        page.goto(url)
        page.wait_for_selector("[data-testid='tweet']", timeout=TIMEOUT)

        # find all tweet background requests:
        tweet_calls = [f for f in _xhr_calls if "TweetResultByRestId" in f.url]
        for xhr in tweet_calls:
            data = xhr.json()
            return data["data"]["tweetResult"]["result"]


async def scrape_tweet_async(_: BeautifulSoup, url: str) -> dict:
    _xhr_calls = []

    def intercept_response(response):
        """capture all background requests and save them"""
        # we can extract details from background requests
        if response.request.resource_type == "xhr":
            _xhr_calls.append(response)
        return response

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        # enable background request intercepting:
        page.on("response", intercept_response)
        await page.goto(url)
        await page.wait_for_selector("[data-testid='tweet']", timeout=TIMEOUT)

        # find all tweet background requests:
        tweet_calls = [f for f in _xhr_calls if "TweetResultByRestId" in f.url]
        for xhr in tweet_calls:
            data = await xhr.json()
            return data["data"]["tweetResult"]["result"]


def scrape_tweets(urls: list[str]) -> list[dict]:
    """
    Scrape multiple tweets from a list of tweet urls.
    This may be very slow.
    """
    _xhr_calls = []

    def intercept_response(response):
        """capture all background requests and save them"""
        # we can extract details from background requests
        if response.request.resource_type == "xhr":
            _xhr_calls.append(response)
        return response

    all_tweets = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # enable background request intercepting:
        page.on("response", intercept_response)

        for url in urls:
            page.goto(url)
            page.wait_for_selector("[data-testid='tweet']", timeout=TIMEOUT)

            # find all tweet background requests:
            tweet_calls = [f for f in _xhr_calls if "TweetResultByRestId" in f.url]
            for xhr in tweet_calls:
                data = xhr.json()
                all_tweets.append(data["data"]["tweetResult"]["result"])

    return all_tweets


def scrape_profile(url: str) -> dict:
    """
    Scrape a X.com profile details e.g.: https://x.com/Scrapfly_dev
    """
    _xhr_calls = []

    def intercept_response(response):
        """capture all background requests and save them"""
        # we can extract details from background requests
        if response.request.resource_type == "xhr":
            _xhr_calls.append(response)
        return response

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # enable background request intercepting:
        page.on("response", intercept_response)
        # go to url and wait for the page to load
        page.goto(url)
        page.wait_for_selector("[data-testid='primaryColumn']")

        # find all tweet background requests:
        profile_data = {}
        tweet_calls = [f for f in _xhr_calls if "UserBy" in f.url]
        for xhr in tweet_calls:
            data = xhr.json()
            profile_data = data["data"]["user"]["result"]

        return profile_data


TWITTER_SCRAPERS: list = [scrape_tweet_async]

if __name__ == "__main__":
    url = "https://x.com/sepiatune/status/1828124393245172094"
    data = scrape_tweet(url)
    print(json.dumps(data, indent=2))
