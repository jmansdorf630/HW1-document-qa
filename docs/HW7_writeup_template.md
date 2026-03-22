# HW7 – Write-up (template)

Fill this in for your submission.

## 1. Solution (architecture)

- **Data:** CSV columns used: …
- **Offline index:** `scripts/build_news_index.py` chunks long `Document` fields, embeds with OpenAI `text-embedding-3-small`, stores in Chroma at `chroma_news_hw7/` collection `NewsHW7`.
- **App:** `HW/HW7.py` loads the collection at startup (no re-embedding of the CSV).
- **Retrieval:**
  - *Topic questions:* `search_news(query)` — vector similarity over chunks.
  - *“Most interesting”:* `retrieve_interesting_candidates()` — broad semantic query, dedupe by `article_id`, rank by distance, then LLM explains why each is interesting (only from excerpts).
- **Grounding:** System prompt restricts answers to retrieved excerpts; URLs/dates in metadata.

## 2. How I tested whether ranked “interesting” results are good

- Manual queries: e.g. “Find the most interesting news”, “Find news about Toyota”, …
- Checked: top items match themes in excerpts; no URLs invented; …
- (Optional) Compared overlap of top-5 between `gpt-4o-mini` and `gpt-4o` for the same retrieval.

## 3. LLM comparison (low vs high cost)

| Aspect        | gpt-4o-mini | gpt-4o |
|---------------|-------------|--------|
| Cost / speed  |             |        |
| Ranking clarity |           |        |
| Faithfulness to excerpts | |        |
| Notes         |             |        |

**Summary (2–3 sentences):** …
