"""EmbeddingGemma 2によるWikipedia本文・画像検索。実行: python experiment.py"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
MODEL = "google/embeddinggemma-2"
K_LIST = (1, 5, 10)


def recall(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def load_data():
    corpus = []
    for path in sorted((BASE / "data").glob("[0-9][0-9][0-9].json")):
        article = json.loads(path.read_text(encoding="utf-8"))
        image = BASE / "data" / article["image_path"]
        for modality, content in (("text", article["text"]), ("image", str(image))):
            corpus.append(dict(
                id=f"{path.stem}:{modality}", title=article["title"],
                modality=modality, content=content
            ))
    queries = json.loads((BASE / "queries.json").read_text(encoding="utf-8"))
    return corpus, queries


def main():
    corpus, queries = load_data()
    model = SentenceTransformer(
        MODEL, config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": torch.float32}
    )
    print(f"model={MODEL}, device={model.device}")
    text_items = [item for item in corpus if item["modality"] == "text"]
    image_items = [item for item in corpus if item["modality"] == "image"]

    images = []
    for item in image_items:
        with Image.open(item["content"]) as image:
            images.append({"image": image.convert("RGB")})
    options = dict(
        batch_size=4, normalize_embeddings=True, show_progress_bar=True,
        convert_to_numpy=True
    )
    documents = [f"title: {item['title']} | text: {item['content']}" for item in text_items]
    text_vectors = model.encode(documents, prompt="", **options)
    image_vectors = model.encode(images, prompt="", **options)
    query_vectors = model.encode([q["query"] for q in queries], prompt_name="SearchQuery", **options)
    corpus = text_items + image_items
    vectors = np.concatenate([text_vectors, image_vectors])
    scores = query_vectors @ vectors.T
    results, summary = [], {}
    for condition in ("text", "image", "mixed"):
        indices = [i for i, item in enumerate(corpus) if condition == "mixed" or item["modality"] == condition]
        candidates = {corpus[i]["id"] for i in indices}
        evaluations = []
        for q, similarities in zip(queries, scores):
            relevant = set(q["relevant"]) & candidates
            if not relevant:
                continue
            order = sorted(indices, key=lambda i: (-float(similarities[i]), corpus[i]["id"]))
            ranked = [corpus[i]["id"] for i in order]
            metrics = {f"Recall@{k}": recall(ranked, relevant, k) for k in K_LIST}
            evaluations.append(metrics)
            top10 = [
                dict(id=corpus[i]["id"], title=corpus[i]["title"], score=float(similarities[i]))
                for i in order[:10]
            ]
            results.append(dict(
                condition=condition, query_id=q["id"], query=q["query"],
                relevant=sorted(relevant), metrics=metrics, top10=top10
            ))
        averages = {
            f"Recall@{k}": sum(m[f"Recall@{k}"] for m in evaluations) / len(evaluations)
            if evaluations else None
            for k in K_LIST
        }
        summary[condition] = dict(candidates=len(indices), queries=len(evaluations), **averages)
    report = dict(
        model=MODEL, device=str(model.device), dtype="float32",
        dimension=int(vectors.shape[1]), summary=summary, results=results
    )
    (BASE / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("詳細: results.json")


if __name__ == "__main__":
    main()
