# Data for HW7

Place your **`news.csv`** here (same columns as the assignment: `company_name`, `days_since_2000`, `Date`, `Document`, `URL`).

Then build the vector index once (from project root):

```bash
export OPENAI_API_KEY=sk-...
python scripts/build_news_index.py
```

Use `--reset` to rebuild after changing the CSV. The app loads the index from `chroma_news_hw7/` at startup.
