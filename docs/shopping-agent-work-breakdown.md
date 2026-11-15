# Autonomous Shopping Agent — Work Breakdown

Based on System Architecture Version 3.1, `shopping-agent-architecture.md`. Section numbers in brackets refer to that document.

## Workstreams

| Code | Workstream | Owns |
|---|---|---|
| DATA | Data Engineering & Ingestion | Adapters, cleaning, normalization, validation, Parquet, geocoding, quality report |
| SEM | Product Semantics & Domain Rules | Category profiles, default and missing-value rules, quantity and overbuy rules |
| SRCH | Search & Retrieval | BM25, embeddings, ONNX model, hybrid ranking, the `search()` interface |
| AGENT | Agent Orchestration & Parsing | LangGraph graph, Pydantic contracts, LLM parse, Clarify, API endpoints |
| DEC | Decision & Optimization Engine | Cost matrix, store selection, enumeration, options, savings and explanation, the `decide()` interface |
| PLAT | Frontend, Infrastructure & MLOps | React UI, Cloud Run, GitHub Actions, Langfuse, evaluation harness, cost controls |

A workstream is listed under a phase only when it has work in that phase.

## Phase 0 — Gate

No production code. The outcome is a go/no-go on the data source and a written semantics spec.

### DATA

**Scope:** decide whether the Kaggle dataset is a sound source and what it contains. Output: a short findings note used by every other workstream.

- Check the Kaggle dataset's update cadence and completeness.
- Check its licence and terms of use.
- Confirm whether promotion files and store files are included.
- Confirm the dataset contains Yesh Hesed, Osher Ad and Netiv HaHesed, with their Bnei Brak branches, and note how each chain is identified in the files.
- Confirm whether the source files carry a category column.
- Run a half-day spike on one or two chains' official price files and compare the effort with the Kaggle CSVs.
- Measure the dataset's real lag, as input for tuning the 7-day staleness cutoff [4.2].
- Start the first per-chain adapters in parallel; they do not depend on the semantics spec.

### SEM

**Scope:** write the one-page product semantics spec [3]. Input: real product names from DATA. Output: the spec that SRCH, AGENT and DEC implement in Phase 1.

- Draft the category profile file with 35 core profiles covering high-frequency household staples: keywords, exclude keywords, defaults, free attributes, typical pack, and typical unit weight for weighed categories [3.1].
- Validate the first-content-word category rule against real product names [3.1].
- Write the two missing-value rules: missing values pass defaults and fail user-stated attributes [3.1].
- Write the quantity rules for the five request forms, the overbuy limit and the small-pack exclusion [3.2].
- State the single ranking rule: total cost to cover the request, unit price as tie-breaker [3.2].
- State the definition of "cheapest" per store count [3.3].

## Phase 1 — Core MVP

Three chains in one city: Yesh Hesed, Osher Ad and Netiv HaHesed, in Bnei Brak. Focus: data reliability, matching accuracy, mathematical correctness.

### DATA

**Scope:** turn the daily Kaggle CSVs into a validated, versioned canonical dataset. Input: Kaggle CSVs, the SEM profile file. Output: canonical Parquet (`products`, `prices`, `stores`, `ingestion_runs`) for SRCH, DEC and PLAT.

- Build three per-chain adapters: Yesh Hesed, Osher Ad and Netiv HaHesed [4].
- Filter stores, and the prices that belong to them, to the Bnei Brak branches of the three chains.
- Implement the canonical schema, including freshness columns and `price_flag` [4.1].
- Build `product_key`: shared across chains only for codes with a valid barcode length and check digit, otherwise `chain:code` [4.1].
- Store prices as integer agorot, per kilogram for weighed items [4.1].
- Partition `prices` by chain through the folder layout [4.1].
- Implement the cleaning rules: duplicates, zero or negative prices, implausible prices carried forward with the original `last_seen_date`, large plausible changes, staleness cutoff [4.2].
- Implement unit normalization with multi-pack expansion [4.3].
- Implement Hebrew text normalization, without prefix stripping [4.4].
- Implement attribute extraction with regex and rules: brand, fat %, volume, weight, flavor [4].
- Apply SEM's category assignment rule at ingestion [3.1].
- Add Pandera schema validation that fails loudly on drift [4].
- Write each run to a dated folder in R2 [4.6].
- Implement geocoding: new or changed stores only, cached, identifying `User-Agent`, one request per second, failure log, `geocode_overrides.csv`, exclusion of stores without coordinates [4.7].
- Prioritize `geocode_overrides.csv` entries for the Bnei Brak branches, checking every branch's coordinates by hand before the demo [4.7].
- Produce the data-quality report on every run [4.8].

### SEM

**Scope:** provide the domain rules as shared code. Input: the Phase 0 spec. Output: a rules library called by DATA (category assignment), SRCH (defaults, filters) and DEC (candidate cost).

- Finalize and maintain the category profile file of 35 core profiles [3.1].
- Implement the category assignment function for products and for list items [3.1].
- Implement default application with the two missing-value rules [3.1].
- Implement the candidate cost function for the five request forms [3.2].
- Implement the per-candidate overbuy filter and the small-pack exclusion [3.2].
- Implement the ranking rule with the unit-price tie-breaker [3.2].
- Author and maintain the labelled evaluation set of 150 items: 60 profile staples, 40 non-profile items, 30 weighed or non-barcode items, and 20 edge cases (exclusions, multi-packs, the 1.5× overbuy limit) [7].
- Hand the set to PLAT in the format the CI harness reads.

### SRCH

**Scope:** return the equivalence class of valid products for each parsed item. Input: canonical Parquet from DATA, parsed items from AGENT, rules from SEM. Output: candidates per item for DEC, clarify signals for AGENT, the index files for PLAT's image build.

- Export `multilingual-e5-small` to an int8-quantized ONNX model [2, 6].
- Precompute product vectors offline with the `passage:` prefix and store them as a float32 NumPy matrix [5.3, 6].
- Build the BM25 index over word tokens and character n-grams [2].
- Add the index build as a pipeline step after canonical Parquet [4].
- Implement query embedding in ONNX Runtime with the `query:` prefix [5.3].
- Implement the `search()` interface in the documented order: profile match and defaults, structured mask over the whole catalog, BM25 and embedding ranking, reciprocal rank fusion, top 200, per-candidate quantity filter [5.3].
- Record candidate IDs, scores and the final choice in the trace [5.3].
- Expose the signals the three clarify rules need [5.4].

### AGENT

**Scope:** orchestrate the request from text to result, and own the API. Input: `list_text` or `parsed_items`, location. Output: a `basket` result or a `clarify` response.

- Define Pydantic models for the graph state and every node's input and output [2, 5.6].
- Define the parsed item schema: `count`, `pack_size`, `total_quantity`, attributes with `strict` [5.2].
- Implement Parse with one Claude Haiku 5.5 structured-output call, with caching [5.2].
- Implement the parse rules for size with and without a count [5.2].
- Implement the rule-based fallback parser, including the `%` marker [5.2].
- Build the six-node LangGraph graph without a checkpointer [5].
- Implement entry at Search when `parsed_items` is submitted, skipping the LLM call [5.1].
- Implement stateless Clarify: the three rules, all questions returned together [5.1, 5.4].
- Handle a resubmitted product that no longer exists: re-run Search and ask again, never substitute silently [5.1].
- Implement `POST /basket` and `GET /health`, both returning `dataset_version` [5.1].
- Implement the failure table and the single error handler with trace ID [5.6].
- Mark unmatched and unfulfilled items as normal result fields [5.6].

### DEC

**Scope:** choose the basket and store split, and explain it. Input: equivalence classes from SRCH, `prices` and `stores` from DATA, cost function from SEM, location. Output: up to three options with savings and explanation, behind the `decide()` interface.

- Select stores in range: radius, nearest 10 per chain, 40 overall filled chain by chain [5.5].
- Build the `cost[i,s]` matrix with DuckDB, using only listed products [5.5, 4.5].
- Implement Haversine distance and the shortest tour over at most six visit orders [5.5].
- Implement subset enumeration for one, two and three stores in NumPy [5.5].
- Discard subsets with an idle store [5.5].
- Apply `F` and `c` from configuration, with travel cost rounded once to agorot [1, 5.5].
- Produce one option per store count, drop dominated options, and mark the recommended one [5.5].
- Compute savings: multi-store against one-store; one-store against the median comparable single store [5.5].
- Write the Explain template, including the sentence on why a fuller basket ranks first [5.5].
- Handle "no stores in range" and "no item listed anywhere" [5.6].
- Write the hand-computed test cases and the two invariant tests [7].

### PLAT

**Scope:** the user interface, the deployment pipeline, monitoring and the evaluation harness. Input: the API from AGENT, the dataset and index files from DATA and SRCH. Output: a running, observable demo.

*Frontend*

- Build the React (Vite) app and host it on Cloudflare Pages or Vercel [2].
- Build the list input and the location step: browser geolocation with a map-pin fallback, with the map centred on Bnei Brak [4.7].
- Build the clarify screen, with all questions on one screen [5.1].
- Build the results view: basket options per store count, items per store, unfulfilled items, a Leaflet map [5.5].
- Add the labels: "listed in the latest price file", "estimate", "simplified parsing, please check" [4.5, 3.2, 5.2].
- State in the UI that Phase 1 uses shelf prices only [10].
- Show the map and geocoding attribution [4.7].

*Infrastructure*

- Write the container image with the dataset as the last layer [4.6].
- Build the GitHub Actions daily workflow: ingest, validate, build the image tagged with the dataset version, deploy a new Cloud Run revision only if validation passed [4.6].
- Configure deployment through Workload Identity Federation [4.6].
- Configure Cloud Run: 1 GB RAM, concurrency 5, maximum 2 instances [1].
- Set up the R2 archive bucket and a registry cleanup policy [2, 9].
- Set the Google Cloud billing alert and the LLM spend cap [7].

*Monitoring and evaluation*

- Integrate Langfuse per-node traces and choose the data region [7].
- Implement the log and trace allowlist, location rounding to about 1 km, and no IP logging [7].
- Track the clarify rate and the "no candidate passes" rate per category [7, 10].
- Build and own the automated CI evaluation harness: parse accuracy, Precision@1, Recall@K before and after filtering, reported per slice of SEM's 150-item set [7].
- Run the cold-start and memory benchmark under five concurrent requests in the first week, on the three-chain Bnei Brak dataset [6, 7].
- Run the ten-basket end-to-end check before the demo, using Bnei Brak locations and the three chains' price files [7].

### Phase 1 sequencing

| Order | Work | Depends on |
|---|---|---|
| 1 | DATA canonical schema; AGENT contracts; PLAT pipeline skeleton and memory benchmark | Phase 0 |
| 2 | SEM rules library; SRCH index build | Canonical schema |
| 3 | SRCH `search()`; DEC cost matrix and enumeration; AGENT Parse | Rules library, contracts |
| 4 | AGENT graph and Clarify; PLAT frontend | `search()`, `decide()` |
| 5 | PLAT evaluation runs and end-to-end check | Everything above |

## Phase 2 — Full coverage and simple promotions

### DATA

**Scope:** extend the dataset to all chains and add simple discounts. Output: the same canonical schema, wider and with discounted prices.

- Add adapters for all remaining chains.
- Ingest unconditional single-item discounts; exclude club-only promotions at first.
- Add the cross-chain barcode mismatch report.

### SRCH

**Scope:** keep retrieval within the memory budget at the full catalog.

- Move the embedding matrix to float16 with chunked casting [6].
- Re-measure the memory budget with PLAT [6].

### PLAT

**Scope:** make data quality visible at full coverage.

- Build the data-quality dashboard from `ingestion_runs` and the quality report.
- Re-run the memory benchmark at the full catalog.

## Phase 3 — Better matching

### SRCH

**Scope:** improve ranking quality using the Phase 1 trace data.

- Train a reranker (LightGBM or a small cross-encoder) on the logged candidates and choices.
- Add the reranker behind the `search()` interface.
- Return two tiers per item from `search()`: exact equivalents, and near substitutes produced by relaxing one user-specified attribute while category defaults stay hard filters.
- Calibrate numeric confidence thresholds for Clarify from the evaluation set [5.4].

### SEM

**Scope:** define behaviour for flexible requests.

- Define soft-preference handling through `strict: false` [5.2].
- Specify the two-tier equivalence model: exact equivalence matches all strict user-specified attributes and category defaults; a near substitute relaxes a single user-specified attribute, such as brand or fat percentage.
- Specify which attributes may be relaxed, one at a time, per category profile.
- Add substitution cases to the labelled evaluation set.

### AGENT

**Scope:** carry soft preferences through the existing schema.

- Have Parse emit `strict: false` where the user expresses a preference.
- Apply the calibrated thresholds in Clarify.
- Carry suggested substitutes in the result as a separate field, and accept the user's confirmation through the existing resubmit path [5.1].

### DEC

**Scope:** decide when a near substitute is offered. Input: both tiers from SRCH. Output: suggested substitutes attached to an option, never included in its total without confirmation.

- Build costs for near substitutes alongside the exact `cost[i,s]` matrix.
- Offer a near substitute only when no exact match is listed in the option's store subset, or when every exact match fails the 1.5× overbuy rule.
- Keep the item unfulfilled in the option's ranking and total until the user confirms the substitute.

### PLAT

**Scope:** show substitutes without substituting silently.

- Show suggested substitutes in a separate row, labelled as an optional alternative that requires confirmation.
- Report substitution cases as their own slice in the CI harness.

## Phase 4 — Multi-buy promotions

### DATA

**Scope:** supply multi-buy promotion data to the decision engine.

- Ingest multi-buy promotions such as "3 for 20" and "2nd at 50%".

### DEC

**Scope:** price baskets where promotions link items within a store.

- Implement a CP-SAT model behind the `decide()` interface.
- Model multi-buy promotions in it.
- Evaluate basket-threshold promotions last; they may not be worth building.

## Phase 5 — Real travel cost

### DEC

**Scope:** replace the distance source only; the per-subset tour already exists.

- Integrate road distance from OSRM or OpenRouteService in place of Haversine.

### AGENT

**Scope:** accept user settings in the API.

- Accept the maximum number of stores alongside the existing `radius_km`.

### PLAT

**Scope:** expose the settings in the UI.

- Add user settings for radius and maximum stores.

## Phase 6 — Public beta

### PLAT

**Scope:** make the system safe and usable for real users.

- Add Supabase accounts, saved lists and history.
- Add rate limiting.
- Run load testing, including DuckDB under multi-user load.
- Tune DuckDB concurrency.
- Define the trace retention period.
- Produce the full cost projection.

### DATA

**Scope:** close the licensing question before opening to the public.

- Confirm the data source licence.

## Phase 7 — Live data

### DATA

**Scope:** replace the Kaggle adapter with the official source, and split the pipeline into a daily catalog build and an hourly price refresh. Output: the daily canonical dataset as before, plus an hourly `prices.parquet` in R2.

- Implement direct hourly pulls from the chains' official files.
- Retire the Kaggle adapter.
- Keep the product catalog, BM25 index and embedding matrix on the daily build.
- Build the hourly price job: pull prices, apply the same cleaning and validation rules, write `prices.parquet`, and upload it to R2 only if validation passed.
- Define what happens to a price whose product is not yet in the daily catalog.

### AGENT

**Scope:** keep each request on one consistent price file, and expose alerts through the API.

- Report both the catalog version and the price version wherever `dataset_version` is returned [5.1].
- Ensure a request reads one price file from start to finish.
- Add endpoints to create, list and dismiss price alerts for a signed-in user.

### DEC

**Scope:** evaluate alerts with the existing decision engine.

- Write the alert evaluation script: after every hourly price update, it checks all active alerts.
- For each alert, recompute the basket for its saved items, location and radius through `decide()`.
- Record every triggered alert in the database, with the time, the new basket total and the price version, and update the last alerted timestamp.
- Fire when the total basket cost is at least 5% or at least 10 NIS below the baseline price.
- Apply the 48-hour cooldown using the last alerted timestamp.

### PLAT

**Scope:** run the hourly refresh in production, and deliver price alerts on-site.

*Hourly price refresh*

- Add the hourly GitHub Actions workflow that runs DATA's price job; the daily image build continues unchanged.
- Have the Cloud Run service download the updated `prices.parquet` from R2 to `/tmp` and atomically point DuckDB to it.
- Re-measure the memory budget, because files in `/tmp` on Cloud Run count against instance memory [6].
- Monitor the age of the price file in use.

*Price alerts*

- Create the `price_alerts` table in Supabase: user ID, shopping list items, location and radius, baseline basket price, last alerted timestamp.
- Add storage in Supabase for triggered alerts, so the history is preserved until the user visits.
- Run the alert evaluation script as a step in the hourly workflow, after the new price file is live.
- Monitor each run: alerts checked, alerts triggered, failures.
- Build the UI to subscribe a saved list to an alert.
- Show the recorded triggered alerts in the web app's notification area or banner when the user visits, including ones triggered days earlier; no email or external push.

This phase may move earlier if the Phase 0 spike shows the official files are easy to use.

## Resolved Architectural Decisions

These five points were open in Version 3 of the architecture and are now decided. The tasks above reflect them.

| Decision | Phase | Summary |
|---|---|---|
| MVP scope | 1 | Bnei Brak, with three chains: Yesh Hesed, Osher Ad and Netiv HaHesed. Adapters, store filtering and geocoding overrides target these branches. |
| Profiles and evaluation set | 1 | 35 core category profiles. 150 labelled items: 60 profile staples, 40 non-profile items, 30 weighed or non-barcode items, 20 edge cases. SEM owns the profiles and the labelled cases; PLAT owns the CI harness. |
| Substitution rules | 3 | Two tiers: exact equivalence, and near substitutes that relax one user-specified attribute while category defaults stay hard. Near substitutes are offered only when no exact match is listed in the store subset or every exact match fails the 1.5× overbuy rule. They appear in a separate row and need the user's confirmation. |
| Hourly data cadence | 7 | Catalog, BM25 index and embedding matrix stay in the daily image. Only `prices.parquet` is refreshed hourly: GitHub Actions uploads it to R2, and the service downloads it to `/tmp` and atomically repoints DuckDB. |
| Price alerts | 7 | Subscriptions live in a Supabase `price_alerts` table. An alert fires when the basket total is at least 5% or at least 10 NIS below the baseline, with a 48-hour cooldown. A script evaluates all active alerts after every hourly price update and records the triggered ones in the database. Delivery is on-site only, from that recorded history. |

The architecture document records these decisions in Version 3.1: scope in section 1, substitutions in 3.4, the hourly price refresh in 4.9, the evaluation set in 7, and price alerts in 11.
