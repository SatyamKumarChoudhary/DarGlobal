# Decision log (ADR style)

## 1. FastAPI + vanilla JS UI instead of Gradio/Streamlit

**Context.** The assignment needs a deployed chatbot with a usable UI.
**Decision.** FastAPI with a hand-written single-page UI served from the same container.
**Consequences.** Full control over SSE streaming, citations UI and branding; auto
OpenAPI docs at `/docs`; slightly more code than Gradio, but the API is a clean,
versioned product on its own, which matches the "production quality" brief.

## 2. Local fastembed embeddings instead of API embeddings

**Context.** Embeddings are needed at build time (indexing) and query time.
**Decision.** `fastembed` with `BAAI/bge-small-en-v1.5` (ONNX, 384-dim).
**Consequences.** Zero API cost and no key needed for ingestion; small image (no
PyTorch); the model is baked into the Docker image so cold starts don't download
anything. Trade-off: English-centric embeddings (see ADR 8).

## 3. FAISS + BM25 hybrid instead of a hosted vector DB

**Context.** The corpus is a few hundred documents.
**Decision.** `faiss-cpu` `IndexFlatIP` (exact cosine) plus `rank-bm25`, fused with
reciprocal rank fusion.
**Consequences.** No external service, exact search (no ANN recall loss at this
scale), and BM25 catches exact project names ("AIDA", "Trump Mansions") that
embeddings may miss. RRF avoids score calibration between the two systems.

## 4. Index built at image build time

**Context.** The runtime must start fast and never scrape or embed on boot.
**Decision.** `uv run property-assistant ingest` runs in the Docker builder stage;
the index and the embedding model cache are copied into the runtime stage.
**Consequences.** Reproducible deploys pinned to the committed data snapshot;
container start is instant; re-indexing requires a rebuild (acceptable: the data
snapshot is versioned in git).

## 5. In-memory sessions with `--workers 1`

**Context.** Conversations need short-lived memory; the free hosting tier is one
small instance.
**Decision.** `InMemoryConversationStore` (TTL + bounded deque) behind a
`ConversationStore` protocol, and a single uvicorn worker.
**Consequences.** Zero infrastructure; horizontal scaling requires the documented
Redis implementation of the same protocol. For this assignment's traffic, one
worker is plenty.

## 6. Hugging Face Spaces (Docker SDK) as primary deploy target

**Context.** The solution must be publicly reachable for free.
**Decision.** HF Spaces Docker SDK (port 7860, uid-1000 user); Render.com as backup.
**Consequences.** Free HTTPS URL and simple git-push deploys. Free Spaces sleep
when idle — the first request after a sleep takes ~30 s (noted in the README and
submission email).

## 7. Free-model fallback strategy

**Context.** Free OpenRouter model IDs change often and have tight rate limits.
**Decision.** Configure `LLM_PRIMARY_MODEL` + `LLM_FALLBACK_MODELS` (validated to
be `:free`-only), send OpenRouter's native `models` array **and** iterate
client-side; cache identical standalone questions; skip the LLM entirely for
ungrounded queries and for first-message query rewriting; per-IP API rate limit.
**Consequences.** The app keeps answering when a model disappears or a quota
trips; worst case it degrades to returning the most relevant source links instead
of a 500.

## 8. English-only scraping scope

**Context.** Both sites publish Arabic variants.
**Decision.** Scrape English pages only (`/en/...` on Wasalt, skip `/ar` on
DarGlobal).
**Consequences.** Halves crawl time and index size; the bge-small-en embedding
model matches the corpus. Arabic questions still get Arabic answers (the LLM
translates), grounded in English sources. Arabic embeddings are listed as future
work.

## 9. Playwright as a scrape-time-only fetcher

**Context.** The spec assumed plain HTTP scraping with `httpx`, with Playwright as
an optional fallback. In practice **both** sites sit behind bot protection
(Incapsula on darglobal.co.uk, Cloudflare on wasalt.sa) that blocks plain HTTP
clients outright.
**Decision.** A `BrowserFetcher` (headless Chromium) implements the same `Fetcher`
protocol as `HttpFetcher` and is used by the scrape CLI. Playwright lives in a
separate `scrape` dependency group and is **not** installed in the Docker image.
Scraping still honours robots.txt and the rate limit; the committed JSONL
snapshot keeps the Docker build reproducible without any scraping.
**Consequences.** The scraper works against the real sites; the runtime image
stays slim; the `HttpFetcher` remains for any future site without bot protection.
