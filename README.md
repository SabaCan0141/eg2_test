# EmbeddingGemma2 精度テスト

## 環境構築

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U -r requirements.txt
```

## Wikipedia収集

```sh
python collect_wikipedia.py
```

実行時に連絡先（WebサイトURL・Wikimedia利用者ページURL・メールアドレスのいずれか）を入力する。

## 検索実験

```sh
python experiment.py
```
