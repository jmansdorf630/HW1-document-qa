# Data for HW7

Place your **`news.csv`** here (same columns as the assignment: `company_name`, `days_since_2000`, `Date`, `Document`, `URL`).

**Streamlit Cloud:** Commit `news.csv` here and set the `openai_api_key` secret. On first load, HW7 **builds the index automatically** if `chroma_news_hw7/` is not present (Cloud has no local pre-built index).

**Local (optional, faster repeat runs):** build the vector index once from project root:

```bash
export OPENAI_API_KEY=sk-...
python scripts/build_news_index.py
```

Use `--reset` to rebuild after changing the CSV. The app loads the index from `chroma_news_hw7/` when it exists.
