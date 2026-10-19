import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from tqdm import tqdm

BASE = Path(__file__).resolve().parent


def read_articles():
    rows, category = [], None
    for line in (BASE / "articles.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            category = line[1:-1].strip()
        else:
            rows.append((category, line))
    return rows


def get(url, contact):
    time.sleep(0.35)
    request = Request(url, headers={"User-Agent": f"Lecture01Collector/1.0 ({contact})"})
    with urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get_content_type()


def query(contact, **params):
    url = "https://ja.wikipedia.org/w/api.php?" + urlencode(
        dict(action="query", format="json", formatversion=2, maxlag=5, **params))
    data = json.loads(get(url, contact)[0])
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["query"]["pages"][0]


def main():
    rows = read_articles()
    contact = input("連絡先（Webサイト・Wikipedia利用者ページURL・メールのいずれか）: ").strip()
    output = BASE / "data"
    output.mkdir()  # 既存データがある場合は上書きせず停止。
    for i, (category, title) in enumerate(tqdm(rows, desc="Wikipedia収集", unit="記事"), 1):
        page = query(
            contact, titles=title, redirects=1, prop="extracts|pageimages|info",
            exintro=1, explaintext=1, piprop="name|thumbnail", pithumbsize=640,
            pilicense="free", inprop="url")
        source = query(
            contact, titles="File:" + page["pageimage"], prop="imageinfo",
            iiprop="url|extmetadata")["imageinfo"][0]
        image, mime = get(page["thumbnail"]["source"], contact)
        extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}[mime]
        image_name = f"{i:03d}{extension}"
        (output / image_name).write_bytes(image)
        record = dict(
            category=category, requested_title=title, title=page["title"],
            text=page["extract"], article_url=page["fullurl"],
            image_path=image_name, image_source=source)
        (output / f"{i:03d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)}記事の収集完了: data/")


if __name__ == "__main__":
    main()
