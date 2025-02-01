import re

# file for custom call backs used by the crawler class to handle various
# discoveries and scenaries


# TMP example callback
def parse_product(response) -> None:
    print(f"found product: {response.url}")


# callbacks represent a set of functions to be called
# by the crawler, and apply a specifc operation to a discovered
# url or element within the DOM
# key is a regex instance to match against, value is the function
# to call.
callbacks = {
    # any url that contains "/products/" is a product page
    re.compile(".+/products/.+"): parse_product
}
