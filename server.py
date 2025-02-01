import os

from flask import Flask, Response, render_template

app = Flask(__name__)


# Simple test server to test web crawlser against


def get_file(filename: str) -> str:
    with open(os.path.join("assets", filename), "r") as f:
        content = f.read()
    return content


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/robots.txt")
def robots_txt() -> Response:
    content = get_file("robots.txt")
    return Response(content, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap_xml() -> Response:
    content = get_file("sitemap.xml")
    return Response(content, mimetype="application/xml")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
