# Autonomous Shopping Agent — System Architecture

Version 3.1 · 2026-11-15 · Version 3 plus five resolved decisions: Phase 1 scope, profile and evaluation-set sizes, substitution rules (3.4), hourly price refresh (4.9) and price alerts (11).

## Changes from version 2

| Change | Effect |
|---|---|
| "Cheapest" is defined by a product semantics spec (category defaults, quantity rules, one ranking rule) | Prevents odd answers such as goat milk or a 3 L jug for "milk". |
| The parsed item separates count, pack size and total quantity | "3 liters of milk" and "a 3-liter carton" are different requests. |
| The Decide step uses exact enumeration of store subsets instead of CP-SAT | No solver statuses, time limits or penalty constants. CP-SAT returns in Phase 4. |
| Decide returns the best option for one, two and three stores | The user can skip a rare item instead of driving to an extra store. |
| The dataset is baked into the container image; a new Cloud Run revision is deployed daily | No hot-swap, no in-memory dataset downloads, atomic switch and rollback for free. Holds through Phase 6; Phase 7 adds an hourly price file (4.9). |
| Clarify is stateless | No checkpointer or database in Phase 1. |
| Simple promotions move to Phase 2 | "Cheapest basket" is credible sooner. |
| Phase 0 grows to three tasks | Dataset check, official-feed spike, semantics spec. |

## 1. Product summary

The user writes a shopping list in natural language and gives a location. The system returns the cheapest basket across nearby Israeli supermarkets, split across a small, reasonable number of stores.

- An unspecified item ("milk") resolves to the cheapest product that fits the category's default profile.
- A specified item ("Tnuva milk 3% 1 liter") is matched exactly on the attributes the user gave.
- A small basket stays in one store; a medium basket may be split between two nearby stores.

### Guiding principles

| Principle | What it means here |
|---|---|
| Maximum output at minimum cost | Free tiers, open-source tools, scale-to-zero hosting, file-based storage. |
| MVP first, zero rewrites | Each phase adds a node, an adapter or a table. None changes the canonical schema or the graph's shape. |
| Custom code over LLM | Table normalization, attribute extraction, product matching and the decision itself are deterministic code and embeddings. The LLM is used once per list, for parsing only. |
| Easy to monitor | Every request is traced per node, every ingestion run produces a quality report, and accuracy is measured in CI. |

### Scope decisions

| Question | Decision |
|---|---|
| Scale | Portfolio / academic demo first; small public beta later. |
| Data source | Kaggle CSV snapshot, refreshed daily, subject to the Phase 0 check. |
| Interface | Simple web app. |
| Phase 1 scope | Bnei Brak, with three chains: Yesh Hesed, Osher Ad and Maayan 2000. |
| LLM | Small paid API, one call per shopping list. |

### Non-functional targets and defaults (proposals)

| Item | Value |
|---|---|
| p95 latency, warm instance | under 3 seconds |
| Cold start | under 10 seconds, to be measured |
| Expected concurrency | up to 5 simultaneous requests |
| Cloud Run settings | 1 GB RAM, concurrency 5, maximum 2 instances (a cost ceiling until rate limiting arrives) |
| Default search radius | 5 km |
| Maximum stores per basket | 3 |
| Candidate stores | nearest 10 per chain, 40 overall |
| `F`, penalty per store beyond the first | 1,000 agorot |
| `c`, travel cost | 100 agorot per km |

`F` and `c` are configurable and tuned against the end-to-end baskets.

## 2. Tech stack

| Layer | Choice | Why it is cheap and sufficient |
|---|---|---|
| Frontend | React (Vite) static site on Cloudflare Pages or Vercel; Leaflet + OpenStreetMap | Free hosting, no map API fees. |
| Backend | Python, FastAPI, LangGraph, in one container | A single service; Python fits the data and ML libraries. |
| Hosting | Google Cloud Run | Scales to zero; the free tier covers demo traffic. |
| Price data | Parquet files inside the container image, queried in-process with DuckDB | Files in the image do not count against instance memory; DuckDB reads only what a query needs. |
| Dataset archive | Cloudflare R2, dated folders | History and rollback source; not read at request time. |
| Ingestion and build | GitHub Actions daily cron, Kaggle API, Polars or pandas, Pandera | Free scheduler, no always-on worker. |
| Lexical search | BM25 over word tokens and character n-grams | No service to run; n-grams handle Hebrew prefixes and spelling variants. |
| Semantic search | `multilingual-e5-small`, product vectors precomputed offline, exact cosine search over a NumPy matrix | No vector database. |
| Query embedding | ONNX Runtime with an int8-quantized model | Keeps PyTorch out of the image and the model at about 120 MB. |
| Decision engine | Exact enumeration in NumPy (Phase 1); OR-Tools CP-SAT (Phase 4) | The Phase 1 problem is small enough to solve exactly without a solver. |
| LLM | Claude Haiku 5.5, structured output, cached | One call per list. |
| Contracts | Pydantic models | Typed state and validation at every node boundary. |
| Monitoring | Langfuse (free cloud tier or self-hosted), structured JSON logs | Per-node traces at no cost. |
| App data (Phase 6) | Supabase free tier | Added only when accounts are needed. Also holds price alerts from Phase 7. |

**Why LangGraph:** it is a stated project requirement and a portfolio piece. Without a checkpointer or interrupt, the pipeline is linear and would also work as plain typed functions.

**Upgrade paths:** retrieval sits behind one `search()` interface, with LanceDB as the single named replacement. The decision engine sits behind one `decide()` interface.

## 3. Product semantics

This section is the summary of the one-page spec written in Phase 0.

### 3.1 Category default profiles

A hand-written file holds 35 core profiles covering high-frequency household staples.

```yaml
milk:
  keywords: [חלב]
  exclude_keywords: [סויה, קוקוס, שוקולד, אבקת, עיזים]
  defaults: {animal: cow, flavor: plain, storage: fresh}
  free: [fat_pct, brand]
  typical_pack: {value: 1000, unit: ml}
tomatoes:
  keywords: [עגבניה, עגבניות]
  exclude_keywords: [רסק, מיובשות]
  weighed: true
  typical_unit_weight: {value: 120, unit: g}
```

- **Category assignment:** a product gets a category when a profile keyword is the first content word of its normalized name, after the brand is removed, and no exclude keyword appears. This keeps "milk chocolate", "soy milk" and "tomato paste" out of the milk and tomato categories. The same keywords assign a category to list items at request time. The rule is validated against the real product names in Phase 0, where the presence of a category column in the source files is also checked.
- **Defaults are hard filters:** they apply when the user did not specify that attribute, and the system never picks a product outside them without the user's confirmation through Clarify.
- **Missing values pass defaults:** a default matches when the product's value is missing or equal to the default, and fails only on an explicit different value. Ordinary milk never says "cow" on the label, so requiring the value would remove it.
- **Missing values fail user requests:** when the user states an attribute, a product with no extracted value for it does not match.
- **Free attributes:** any value is accepted, so the cheapest wins.

### 3.2 Quantity rules

The parsed item carries a `count`, and at most one of `pack_size` and `total_quantity` (section 5.2).

| Request form | Example | Candidate cost |
|---|---|---|
| Total quantity | "3 L milk" | `ceil(total ÷ pack size) × pack price`, using packs of one product only |
| Pack size and count | "milk 1 L ×2" | `count × pack price`; the pack size must match |
| Count only | "milk ×2" | `count × pack price`; the pack must be within 0.8–1.5× of the category's typical pack |
| Weighed, by weight | "1.5 kg tomatoes" | Price per kilogram × grams ÷ 1000, rounded once |
| Weighed, by count | "6 tomatoes" | `count × typical_unit_weight` gives the weight; the result is labelled an estimate |

- **One ranking rule:** candidates are ranked by the total cost of covering the request. Unit price is a tie-breaker only.
- **No mixing:** pack sizes of different products are never combined.
- **Small packs:** for a total-quantity request in a category with a profile, packs smaller than 0.8× the typical pack are excluded, so "3 L milk" is not met with twelve small cartons.
- **Overbuy:** a candidate that would deliver more than 1.5× the requested total quantity is filtered out on its own. The item goes to Clarify only when every otherwise-valid candidate fails on overbuy.

### 3.3 Definition of "cheapest"

For each store count (one, two, three), the best basket is the one that fulfils the most items and, among those, has the lowest total of basket price, extra-store penalties and travel cost (section 5.5). The recommended answer is the best across all three; the others are offered as alternatives.

### 3.4 Substitutions (Phase 3)

| Tier | Definition |
|---|---|
| Exact equivalence | Matches all strict user-specified attributes and the category defaults. This is the Phase 1 behaviour. |
| Near substitute | Relaxes a single user-specified attribute, such as brand or fat percentage. Category defaults remain hard filters. |

- **Trigger:** a near substitute is offered only when no exact match is listed in the option's store subset, or when every exact match fails the 1.5× overbuy rule.
- **Never silent:** suggested substitutes appear in a separate row, labelled as an optional alternative that requires the user's confirmation.
- **Totals:** the item stays unfulfilled in the option's ranking and total until the user confirms.

## 4. Data pipeline (offline, daily)

```
Kaggle CSVs
  → per-chain adapters          absorb each chain's inconsistent format
  → cleaning                    dedupe, drop invalid prices, handle implausible prices
  → unit normalization          base quantity and base unit per product
  → text normalization          Hebrew name normalization
  → attribute extraction        brand, fat %, volume, weight, flavor, category (regex and rules)
  → schema validation           Pandera; fail loudly on drift
  → canonical Parquet           written to a new dated folder
  → search index build          BM25 index + embedding matrix
  → data-quality report
  → image build and deploy      only if validation passed
```

### 4.1 Canonical schema

| Table | Key columns |
|---|---|
| `products` | `product_key`, `barcode`, `name`, `name_normalized`, `brand`, `category`, `size_value`, `size_unit`, `base_quantity`, `base_unit`, `is_weighed`, `attributes` |
| `prices` | `product_key`, `store_id`, `price_agorot`, `price_flag`, `last_seen_date`, `source_timestamp`, `ingestion_timestamp`, `dataset_version` |
| `stores` | `store_id`, `chain`, `name`, `address`, `city`, `lat`, `lon`, `geocode_status` |
| `ingestion_runs` | `dataset_version`, `chain`, `validation_status`, `rows_in`, `rows_rejected`, quality metrics |

- **Product key:** a code with a valid barcode length and check digit is shared across chains. Any other code is namespaced as `chain:code` and never joined across chains.
- **Money:** prices are integer agorot. For weighed items the price is per kilogram.
- **Unit price:** not stored. It is computed at query time as a float and used only as a tie-breaker; totals are always count × integer agorot.
- **Partitioning:** `prices` is partitioned by `chain` through the folder layout.
- **Column names** are a proposal, to be adjusted against the actual Kaggle files.

### 4.2 Cleaning

| Case | Action |
|---|---|
| Duplicate barcode within a chain and store | Keep the latest row; count the rest in the report. |
| Zero or negative price | Drop and count. |
| Implausible price (for example a tenfold change, or below a minimum floor) | Carry the previous version's row forward with its original `last_seen_date`, set `price_flag`, and list it in the report. |
| Large but plausible price change | Accept; list in the report. |
| Listing older than the staleness cutoff | Exclude. The cutoff is 7 days, configurable, tuned after Phase 0. |

Carrying the previous row forward keeps the item listed, and the staleness cutoff removes it if the problem persists.

### 4.3 Unit normalization

Every product gets a `base_quantity` in a base unit (millilitres, grams or units). Multi-packs are expanded (6 × 1.5 L = 9,000 ml). This is what makes package sizes comparable.

### 4.4 Hebrew text normalization

- Strip niqqud and punctuation noise.
- Normalize geresh and gershayim.
- Canonicalize unit abbreviations (ל', מ"ל, גר', ק"ג and their variants).
- Prefixes are not stripped, because that would damage real product words such as לחם, מלח, בשר and שמן. Character n-grams in BM25 and the embedding model handle prefixes and spelling variants.

### 4.5 Availability

The price files carry no stock information. An item is either listed in the store's latest file or not. The UI labels this "listed in the latest price file", not "in stock". Items can only be bought where they are listed.

### 4.6 Versioning and deployment

1. Ingestion writes a complete dated folder to R2 (`datasets/v=2026-11-15/`).
2. If validation passes, GitHub Actions builds a container image containing that folder, tagged with the dataset version.
3. The image is deployed as a new Cloud Run revision.

- A request is served entirely by one revision, so it always sees one consistent dataset.
- In-flight requests finish on the old revision.
- Rollback is switching traffic to the previous revision.
- A failed ingestion run deploys nothing.
- The dataset is the last image layer, so daily builds reuse the cached Python and model layers.
- GitHub Actions deploys through Workload Identity Federation, so no service-account key is stored.
- This is the whole serving model through Phase 6. Phase 7 adds an hourly price file on top of it (4.9).

### 4.7 Geocoding

- Only new or changed stores are geocoded; results are cached in `stores`.
- Nominatim is called with an identifying `User-Agent`, at most one request per second, and attribution is shown in the app.
- Failures are logged with the raw address. A manual `geocode_overrides.csv` supplies coordinates for them. Expect this file to need real attention, since Hebrew address quality is uneven.
- Stores without coordinates are excluded and counted in the report.
- The user's location comes from browser geolocation. If the user declines, they drop a pin on the map. Nothing is geocoded at request time.

### 4.8 Data-quality report (every run)

Rows per chain, rejected rows, duplicates removed, missing barcodes and fields, flagged and large price changes, stores without coordinates, and the age of the newest `source_timestamp` per chain.

### 4.9 Hourly price refresh (Phase 7)

Catalog updates are decoupled from price updates, which avoids 24 full image rebuilds a day.

| Artifact | Cadence | Where it lives |
|---|---|---|
| Product catalog, BM25 index, embedding matrix | Daily | Baked into the container image, as in 4.6 |
| `prices.parquet` | Hourly | Uploaded to R2 by GitHub Actions; downloaded by the service |

1. An hourly GitHub Actions job pulls prices, applies the same cleaning and validation rules, and uploads `prices.parquet` to R2 only if validation passed.
2. The Cloud Run service downloads the new file to `/tmp` and atomically points DuckDB to it.
3. A request reads one price file from start to finish.

- Responses report the catalog version and the price version separately.
- Files in `/tmp` on Cloud Run count against instance memory, so the memory budget is re-measured in Phase 7 (section 6).
- The handling of a price whose product is not yet in the daily catalog is defined in Phase 7.

## 5. Request flow (online)

```
shopping list + location
  → 1. Parse      LLM: text → structured items
  → 2. Search     hybrid retrieval + category defaults + hard filters → candidates per item
  → 3. Clarify    returns options to the client when a rule fires (stateless)
  → 4. Compare    DuckDB: cost of each item at each store within the radius
  → 5. Decide     exact enumeration: best basket for one, two and three stores
  → 6. Explain    template: basket per store, total, savings, unfulfilled items
```

Search, Compare and Decide are the three "agents". They are LangGraph nodes with typed inputs and outputs, and only Parse calls an LLM.

### 5.1 API contract

| Endpoint | Input | Output |
|---|---|---|
| `POST /basket` | Either `list_text` or `parsed_items` (with any clarify choices), plus `lat`, `lon`, optional `radius_km` | Either a `basket` result or a `clarify` response; both include `dataset_version` |
| `GET /health` | none | Service status and `dataset_version` |

- A `clarify` response contains the parsed items and the options for every ambiguous item in the list, shown together on one screen.
- The client resubmits `parsed_items` with the user's choices. The graph then enters at Search and skips the LLM call.
- If a chosen product no longer exists because a new dataset was deployed in between, Search is re-run for that item from its text and the user is asked to choose again. A product is never substituted silently.

### 5.2 Parse

One LLM call returns a list of items:

```json
{
  "item": "milk",
  "count": 2,
  "pack_size": {"value": 1000, "unit": "ml"},
  "total_quantity": null,
  "attributes": {
    "brand":   {"value": "Tnuva", "strict": true},
    "fat_pct": {"value": 3,       "strict": true}
  }
}
```

| Input | Parsed as |
|---|---|
| A size without a count ("3 L milk") | `total_quantity` |
| A size with a count, or an explicit container ("milk 1 L ×2", "a 3-liter carton") | `pack_size` and `count` |
| A count only ("milk ×2", "6 tomatoes") | `count` |

- At most one of `pack_size` and `total_quantity` is set. These cases are part of the Parse evaluation set.
- In Phase 1, every attribute the user specifies is strict. The `strict` flag exists from day one, so soft preferences arrive in Phase 3 without a schema change.
- **LLM outage fallback:** a rule-based parser treats each line as one item. A number followed by a unit is a size, using the same unit table as ingestion; a number followed by `%` is an attribute; a bare number is a count. Fallback results are labelled in the UI as "simplified parsing, please check", because a wrong parse that still finds valid products does not trigger Clarify.

### 5.3 Search

1. The item is matched to a category profile by keyword, and the profile's defaults fill the attributes the user left unspecified.
2. Structured hard filters (category, brand, pack size, fat % and other strict attributes) are applied as a mask over the whole catalog, before any ranking cut-off.
3. BM25 and embedding search rank the remaining products. Queries and products use the e5 `query:` and `passage:` prefixes.
4. The two rankings are merged with reciprocal rank fusion, and the top 200 are kept.
5. The quantity rules filter each candidate individually.
6. The survivors form the item's equivalence class.

Candidate IDs, scores and the final choice are recorded in the trace, which later becomes training data for the Phase 3 reranker.

### 5.4 Clarify

The node returns a `clarify` response on any of three rules:

- No candidate passes the hard filters.
- The item matches no category profile and the top candidates differ in attributes other than brand, size and pack.
- Every otherwise-valid candidate fails the overbuy limit.

Numeric confidence thresholds are calibrated from the evaluation set in Phase 3.

### 5.5 Compare and Decide

**Compare** builds a `cost[i,s]` matrix: for each item and each store in range, the cost of the cheapest listed member of the equivalence class under the quantity rules. Cells are empty where nothing is listed.

**Stores in range:** all stores within the radius, capped at the nearest 10 per chain and 40 overall. The overall cap is filled chain by chain in turn (each chain's nearest store, then each chain's second nearest, and so on), so a slightly farther discount chain is not cut out.

**Decide** enumerates every subset of one, two or three stores:

```
for each subset S:
    fulfilled(S) = number of items with at least one non-empty cost in S
    basket(S)    = Σ over fulfilled items of  min over s in S of cost[i,s]
    travel(S)    = c × shortest Haversine tour  home → stores in S → home
    total(S)     = basket(S) + F × (|S| − 1) + travel(S)

for each store count k in {1, 2, 3}:
    option(k) = the subset of size k with the highest fulfilled(S), then the lowest total(S)
```

- **Idle stores:** a subset in which some store supplies no item is discarded, because the same basket is already covered by a smaller subset. Penalties and travel therefore apply only to stores that supply at least one item.
- **Rounding:** distances stay as floating-point kilometres; the travel cost is rounded once to agorot at the end.
- **Options returned:** up to three, one per store count. An option is dropped when one with fewer stores fulfils at least as many items for no higher total.
- **Recommended option:** the most items fulfilled, then the lowest total.
- **Explanation to the user:** the recommended option covers the most items, even when it costs more than an option that skips some; the alternatives show what skipping an item saves.
- **Savings:** a multi-store option is compared with the one-store option, with any difference in fulfilled items stated. A one-store option is compared with the median single store that carries the same number of items; if there is none, no saving is shown.
- **Size:** 30 candidate stores give 4,525 subsets and the cap of 40 gives 10,700, solved exactly in milliseconds with NumPy.
- **Tour:** the shortest of the visit orders; at most six for three stores.

A split is recommended only when it fulfils more items or when the saving exceeds the extra-store penalty plus the extra travel. The user can always choose a fewer-store alternative and skip a rare item.

### 5.6 Validation and failure handling

| Situation | Behaviour |
|---|---|
| Structural or business-rule violation at a node boundary | Pydantic validators raise; the error handler returns a clear message with the trace ID. |
| LLM unavailable | Rule-based parser. |
| Search fails for one item | The item is marked unmatched; the rest of the basket continues. |
| No stores in range | Result says so and suggests a larger radius. |
| No item listed anywhere in range | Result lists all items as unfulfilled. |
| Compare or Decide raises | Error handler; clear message with the trace ID. |

Unmatched and unfulfilled items are normal fields in the result, not errors.

## 6. Memory budget

These are estimates for Phase 1 scope (three chains in Bnei Brak, about 50,000 products). They are verified by a benchmark in the first week of Phase 1.

| Component | Estimate |
|---|---|
| Python, FastAPI, LangGraph | 150–200 MB |
| Embedding model, int8-quantized ONNX | about 120 MB |
| Embedding matrix, float32 | about 80 MB |
| BM25 index | about 50 MB |
| DuckDB working memory | 100–200 MB |
| **Total** | **500–650 MB of 1 GB** |

- The unquantized model is roughly 470 MB by parameter count, so int8 quantization is required.
- Parquet files live in the image and are not loaded into memory wholesale.
- The benchmark measures peak memory with five concurrent requests, not one.
- At the full catalog in Phase 2, the matrix moves to float16 with chunked casting, since NumPy has no fast float16 matrix multiply. The budget is re-measured then.
- In Phase 7 the hourly `prices.parquet` in `/tmp` is added to the budget, and briefly counted twice while a new file replaces the old one.

## 7. Monitoring, evaluation and privacy

| Area | What is measured | When |
|---|---|---|
| Request traces | Per-node latency, inputs, outputs, LLM cost, candidates and choices | Every request |
| Clarify rate | Share of lists and of items that trigger Clarify, per rule | Every request |
| Data quality | The report in section 4.8 | Every ingestion run |
| Parse accuracy | Structured output against a hand-labelled set, including the count, pack size and total quantity cases | CI |
| Product matching | Precision@1, and Recall@K before and after hard filtering, on 150 labelled items, reported per slice | CI |
| Decision correctness | Hand-computed small cases, plus two invariants: adding a store to the candidate set after the caps are applied never worsens the recommended option, and a very large `F` always yields one store | CI |
| End to end | Ten real baskets checked by hand against the price files | Before the Phase 1 demo, then on change |
| Cold start and memory | Startup latency, and peak RAM under five concurrent requests | First week of Phase 1 |
| Load | Latency and concurrency, including DuckDB under multi-user load | Before the beta (Phase 6) |

**Labelled evaluation set (150 items):**

| Slice | Items |
|---|---|
| Profile staples | 60 |
| Non-profile items | 40 |
| Weighed or non-barcode items | 30 |
| Edge cases: exclusions, multi-packs, the 1.5× overbuy limit | 20 |

The Semantics workstream authors and maintains the profiles and the labelled cases. The Platform workstream owns the automated CI harness.

**Privacy:**

- Logs and traces use an allowlist: only named fields are recorded.
- Location is rounded to about 1 km before it is recorded.
- IP addresses are not logged.
- Phase 1 has no accounts, so list text in traces is not tied to an identity.
- The Langfuse data region is chosen at signup. A trace retention period is defined before the beta.

**Cost control:** a billing budget alert on Google Cloud, a monthly spend cap on the LLM key, and the Cloud Run instance limit, all set in Phase 1.

## 8. Phased roadmap

| Phase | Adds | Notes |
|---|---|---|
| 0. Gate | Kaggle check (update cadence, licence, promotion and store files, category column, presence of the three Phase 1 chains and their Bnei Brak branches); half-day spike on one or two chains' official files; one-page semantics spec | Ingestion adapters can start in parallel. |
| 1. Core MVP | Yesh Hesed, Osher Ad and Maayan 2000 in Bnei Brak; full pipeline and request flow as described; web page with list input, clarify choices, basket options and a map; traces; evaluation suites | Focus: data reliability, matching accuracy, mathematical correctness. |
| 2. Full coverage and simple promotions | Adapters for all chains; unconditional single-item discounts; cross-chain barcode mismatch report; float16 matrix; data-quality dashboard | Club-only promotions are excluded at first. |
| 3. Better matching | Trained reranker from Phase 1 logs; soft preferences via `strict: false`; two-tier substitutions; calibrated confidence thresholds | Substitution rules are in 3.4. |
| 4. Multi-buy promotions | "3 for 20", "2nd at 50%"; CP-SAT behind the `decide()` interface | Basket-threshold promotions come last and may not be worth it. |
| 5. Real travel cost | Road distance (OSRM or OpenRouteService) in place of Haversine; user settings for radius and maximum stores | The per-subset tour already exists, so only the distance source changes. |
| 6. Public beta | Supabase accounts, saved lists, history; rate limiting; load testing; DuckDB concurrency tuning; trace retention; full cost projection; licence confirmed | |
| 7. Live data | Direct hourly pulls from the chains' official files; hourly price refresh separate from the daily image (4.9); on-site price alerts evaluated every hour (11) | Replaces the Kaggle adapter. May move earlier if the Phase 0 spike shows the official files are easy to use. |

### Phase 1 checklist

- **Ingestion:** adapters, cleaning, unit and text normalization, attribute and category extraction, barcode check digit, freshness columns, dated folders, geocode cache with override file, quality report, daily image build and deploy.
- **Semantics:** category profile file with keywords, exclude keywords, defaults, typical pack and typical unit weight; first-word category rule; missing-value rule; quantity rules; one ranking rule.
- **Matching:** BM25 with n-grams, NumPy cosine search, int8 ONNX query embedding, pre-filter mask, rank fusion, top 200, `strict` flag, three clarify rules, candidate logging.
- **Decision:** `cost[i,s]` matrix, per-chain and overall store caps, subset enumeration without idle stores, one option per store count, single rounding of travel cost.
- **Graph:** six nodes, Pydantic contracts, stateless Clarify, rule-based parse fallback, one error handler.
- **Frontend:** list input, geolocation with map-pin fallback, clarify choices, basket options, map, "listed" and "estimate" labels, attribution.
- **Operations:** Langfuse traces, log allowlist, location rounding, billing alert, LLM spend cap, Cloud Run limits, memory benchmark.
- **Evaluation:** parse, matching, decision and end-to-end checks.

### Deliberately not in Phase 1

Reranker, soft preferences, substitutions, promotions, road distances, accounts, dashboards, load testing, a vector database, a solver, dataset hot-swap, mixed pack-size combinations, Hebrew prefix stripping, pruning of identically priced stores, direct feeds.

## 9. Estimated cost

These are estimates based on the providers' free tiers, not quotes.

| Item | Demo (Phases 1–5) | Public beta (Phase 6+) |
|---|---|---|
| Frontend hosting | $0 | $0 |
| Backend (Cloud Run) | $0 | $5–15/month |
| Image and archive storage | $0–1/month | $0–3/month |
| LLM | under $5/month | $5–20/month |
| Monitoring (Langfuse) | $0 | $0 |
| Database (Supabase) | not used | $0, or $25/month beyond the free tier |
| Domain (optional) | about $12/year | about $12/year |
| **Total** | **$0–6/month** | **roughly $10–60/month** |

Daily image builds accumulate in the container registry, so a cleanup policy keeps only the last few versions.

## 10. Risks

| Risk | Mitigation |
|---|---|
| The Kaggle dataset stops updating or its licence restricts use | Phase 0 check and official-feed spike; the adapter layer lets the source be replaced. |
| Memory exceeds 1 GB | Quantized model, dataset in the image, written budget, first-week benchmark under concurrency. |
| Shelf prices only in Phase 1 | Stated in the UI; simple promotions arrive in Phase 2. |
| Category profiles miss a common item | Clarify catches it; the profile file grows from trace data. |
| Keyword-based category assignment mislabels products | Measured by the matching evaluation; rules refined per category. |
| Items without shared barcodes (produce, deli) | Name-based matching; tracked as a separate evaluation slice. |
| Attribute regex errors silently empty the candidate set | "No candidate passes" rate is a tracked metric per category. |
| Geocoding quality on Hebrew addresses | Override file; the Bnei Brak branches are checked by hand first. |
| A Phase 1 chain is missing from the dataset or published under a parent chain's files | Phase 0 check of how each of the three chains is identified. |
| Hourly price file pushes memory past 1 GB (Phase 7) | Re-measured budget; only prices are refreshed hourly, and the heavy artifacts stay in the image. |
| Hourly jobs exceed the free GitHub Actions minutes (Phase 7) | Keep the hourly job to prices and alert evaluation only; measure run time before enabling. |

## 11. Price alerts (Phase 7)

Alerts are on-site only: no email and no external push service.

| Aspect | Decision |
|---|---|
| Subscriptions | A `price_alerts` table in Supabase: user ID, shopping list items, location and radius, baseline basket price, last alerted timestamp. |
| Trigger | The total basket cost is at least 5% or at least 10 NIS below the baseline price. |
| Cooldown | 48 hours per alert. |
| Evaluation | Proactive. After every hourly price update, a script checks all active alerts by recomputing each basket through `decide()`. |
| History | Every triggered alert is recorded in the database, so it is preserved even if the user visits days later. |
| Delivery | Recorded alerts are shown in the web app's notification area or banner when the user visits. |
