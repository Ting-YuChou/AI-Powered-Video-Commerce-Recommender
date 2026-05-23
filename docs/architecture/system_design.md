# AI Video Commerce Recommendation System Design

> Format: English first, Traditional Chinese second.
> 格式：英文在前，繁體中文在後。

---

## 1. How I Would Answer This in a System Design Interview

### English
When answering this design in an interview, I would follow this order:

1. Clarify scope and goals
2. Define functional and non-functional requirements
3. Estimate scale
4. Define APIs and core data model
5. Propose a high-level architecture
6. Walk through the main request flows
7. Deep dive into storage, caching, and asynchronous processing
8. Explain scaling and reliability
9. Discuss trade-offs and alternatives
10. Close with what I would build first vs later

### 繁體中文
如果在 system design 面試中回答這題，我會照下面順序講：

1. 先釐清 scope 與目標
2. 定義功能需求與非功能需求
3. 做流量與容量估算
4. 定義 API 與核心資料模型
5. 提出 high-level architecture
6. 走一遍主要 request flow
7. 深入說明 storage、cache 與非同步處理
8. 說明如何擴展與提高可靠性
9. 討論 trade-off 與替代方案
10. 最後補充 first version 跟 future iteration

---

## 2. Problem Statement

### English
Design a recommendation system for a short-video commerce platform. The platform should:

- ingest product videos uploaded by merchants or creators,
- process the content into searchable embeddings,
- collect user interaction events in real time,
- return personalized recommendations with low latency,
- continue to improve recommendations as user behavior changes.

This repository's current implementation already reflects a service-oriented architecture with:

- `gateway-api`
- `recommendation-service`
- `interaction-ingest-service`
- Kafka workers
- Redis feature store
- FAISS vector search
- periodic model retraining

### 繁體中文
要設計一個短影音電商平台的推薦系統。這個平台需要：

- 接收商家或創作者上傳的商品影片，
- 將內容處理成可搜尋的 embedding，
- 即時蒐集使用者互動事件，
- 以低延遲回傳個人化推薦，
- 並且能隨著使用者行為變化持續優化推薦品質。

這個 repo 目前的實作已經是 service-oriented architecture，核心包含：

- `gateway-api`
- `recommendation-service`
- `interaction-ingest-service`
- Kafka workers
- Redis feature store
- FAISS 向量搜尋
- 週期性模型重訓

---

## 3. Scope Clarification

### English
Before proposing the architecture, I would explicitly define what is in scope.

In scope:

- online recommendation serving
- user interaction ingestion
- video upload orchestration
- async content processing
- candidate retrieval and ranking
- basic observability and health checks
- periodic retraining

Out of scope for the first version:

- payments and order management
- merchant CMS workflow
- ad auction system
- cross-region active-active deployment
- strict exactly-once event guarantees
- advanced abuse detection

### 繁體中文
在提出架構前，我會先明確定義哪些內容在 scope 內。

第一版 scope 內：

- 線上推薦服務
- 使用者互動事件接收
- 影片上傳編排
- 非同步內容處理
- 候選召回與排序
- 基本監控與健康檢查
- 週期性模型重訓

第一版 scope 外：

- 金流與訂單管理
- 商家 CMS 工作流
- 廣告競價系統
- 跨區 active-active 部署
- 嚴格 exactly-once 事件保證
- 進階風控與濫用偵測

---

## 4. Requirements

### 4.1 Functional Requirements

#### English

- Users can request recommendations with `user_id`, optional `content_id`, and context.
- Clients can send interaction events such as `view`, `click`, `add_to_cart`, `purchase`, `favorite`, and `share`.
- Clients can upload video content for asynchronous processing.
- The system should support content-based, collaborative, and trending recommendation strategies.
- The system should support cold-start fallback.
- The system should expose health and metrics endpoints.

#### 繁體中文

- 使用者可帶著 `user_id`、可選的 `content_id` 與 context 請求推薦。
- Client 可以上報 `view`、`click`、`add_to_cart`、`purchase`、`favorite`、`share` 等事件。
- Client 可以上傳影片並由系統非同步處理。
- 系統需支援 content-based、collaborative、trending 三種推薦策略。
- 系統需支援 cold-start fallback。
- 系統需提供 health 與 metrics 端點。

### 4.2 Non-Functional Requirements

#### English

- Recommendation latency should be low, with a target such as P99 under 200 ms.
- Interaction ingestion should handle bursty write traffic.
- The system should degrade gracefully if Kafka or a worker is unavailable.
- Recommendation quality should improve as more interactions arrive.
- The architecture should allow independent scaling of serving and ingestion.

#### 繁體中文

- 推薦延遲要低，目標例如 P99 低於 200 ms。
- 互動事件接收要能承受尖峰寫入流量。
- 如果 Kafka 或 worker 不可用，系統要能 graceful degradation。
- 隨著更多互動資料進來，推薦品質應逐步提升。
- 架構要允許推薦服務與事件接收服務獨立擴容。

---

## 5. Capacity Estimation

### English
In an interview, I would make simple assumptions and use them to justify the architecture.

Example assumptions:

- 10 million DAU
- 100 recommendation requests per second average, 10x peak = 1,000 RPS
- 5,000 interaction events per second average, 5x peak = 25,000 EPS
- 100,000 active products
- 10,000 new or updated content uploads per day
- recommendation response payload around a few KB

These assumptions suggest:

- recommendation serving should be separated from ingestion,
- writes should be asynchronous,
- Redis is appropriate for online feature/cache access,
- Kafka is appropriate as the event backbone,
- FAISS or ANN search is needed for low-latency retrieval at scale.

### 繁體中文
在面試中，我會先做簡單假設，再用這些假設來合理化架構選擇。

示意假設：

- 1,000 萬 DAU
- 推薦請求平均每秒 100 次，尖峰 10 倍，約 1,000 RPS
- 互動事件平均每秒 5,000 次，尖峰 5 倍，約 25,000 EPS
- 活躍商品 10 萬個
- 每天 1 萬筆新上傳或更新內容
- 推薦 response 大小約數 KB

這些估算代表：

- 推薦服務應與事件接收拆開，
- 寫入流程應盡量非同步，
- Redis 適合做線上特徵與快取存取，
- Kafka 適合做事件骨幹，
- FAISS 或其他 ANN 搜尋是低延遲召回所需要的。

---

## 6. API Design

### English
I would define the external APIs first, then describe how the gateway routes to internal services.

External API via `gateway-api`:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/recommendations` | return ranked product recommendations |
| `POST` | `/api/interactions` | ingest user interaction asynchronously |
| `POST` | `/api/content/upload` | upload a video for async processing |
| `GET` | `/api/content/{content_id}/status` | check content processing status |
| `GET` | `/api/analytics` | fetch aggregate analytics |
| `GET` | `/health` | aggregated system health |
| `GET` | `/metrics` | Prometheus metrics |

Example recommendation request:

```json
{
  "user_id": "user_123",
  "content_id": "content_456",
  "context": {
    "device": "mobile",
    "session_id": "sess_abc"
  },
  "k": 10
}
```

Example interaction request:

```json
{
  "user_id": "user_123",
  "product_id": "prod_789",
  "action": "click",
  "context": {
    "page": "video_feed",
    "recommendation_position": 2
  }
}
```

### 繁體中文
我會先定義對外 API，再說明 gateway 如何轉發到內部服務。

對外 API 由 `gateway-api` 統一暴露：

| Method | Path | 用途 |
|---|---|---|
| `POST` | `/api/recommendations` | 回傳排序後的商品推薦 |
| `POST` | `/api/interactions` | 非同步接收使用者互動 |
| `POST` | `/api/content/upload` | 上傳影片並進行非同步處理 |
| `GET` | `/api/content/{content_id}/status` | 查詢內容處理狀態 |
| `GET` | `/api/analytics` | 取得聚合分析資料 |
| `GET` | `/health` | 聚合健康檢查 |
| `GET` | `/metrics` | Prometheus 指標 |

推薦請求範例：

```json
{
  "user_id": "user_123",
  "content_id": "content_456",
  "context": {
    "device": "mobile",
    "session_id": "sess_abc"
  },
  "k": 10
}
```

互動事件範例：

```json
{
  "user_id": "user_123",
  "product_id": "prod_789",
  "action": "click",
  "context": {
    "page": "video_feed",
    "recommendation_position": 2
  }
}
```

---

## 7. Core Data Model

### English
At interview level, I would keep the data model simple:

- User Features
  - `user_id`
  - total interactions
  - preferred categories
  - CTR / CVR
  - last active timestamp

- Content Features
  - `content_id`
  - visual embedding
  - detected objects / OCR / metadata
  - processing status

- Product Metadata
  - `product_id`
  - title, category, brand, price, rating

- Interaction Event
  - `user_id`
  - `product_id`
  - `action`
  - `context`
  - `timestamp`

- Recommendation Cache Entry
  - `user_id`
  - context hash
  - ranked recommendations
  - TTL

### 繁體中文
在面試層級上，我會把資料模型講得簡潔但完整：

- User Features
  - `user_id`
  - 總互動數
  - 偏好類別
  - CTR / CVR
  - 最後活躍時間

- Content Features
  - `content_id`
  - 視覺 embedding
  - 偵測物件 / OCR / metadata
  - 處理狀態

- Product Metadata
  - `product_id`
  - title、category、brand、price、rating

- Interaction Event
  - `user_id`
  - `product_id`
  - `action`
  - `context`
  - `timestamp`

- Recommendation Cache Entry
  - `user_id`
  - context hash
  - 排序後推薦結果
  - TTL

---

## 8. High-Level Architecture

### English
I would propose a split architecture with one edge service, two online services, and several asynchronous workers.

```text
Client / Frontend
        |
        v
+----------------------+
| Gateway API          |
| auth / rate limit    |
| routing / upload     |
+----+-------------+---+
     |             |
     |             |
     v             v
+------------------+     +---------------------------+
| Recommendation   |     | Interaction Ingest        |
| Service          |     | Service                   |
+----+---------+---+     +-------------+-------------+
     |         |                       |
     |         v                       v
     |    +---------+            +-----------+
     |    | Redis   |<---------->| Kafka     |
     |    +----+----+            +-----+-----+
     |         |                       |
     v         v                       v
+----------------------+      +----------------------+
| RecommendationEngine |      | Content Worker       |
| RankingModel         |      | Feature Worker       |
| VectorSearch / TT    |      | Model Trainer        |
+----------------------+      +----------------------+
```

This matches the current codebase:

- [`gateway_api.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/gateway_api.py)
- [`recommendation_api.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/recommendation_api.py)
- [`interaction_ingest_api.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/interaction_ingest_api.py)
- [`kafka_workers/video_processor.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/kafka_workers/video_processor.py)
- [`kafka_workers/feature_updater.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/kafka_workers/feature_updater.py)
- [`model_trainer.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/model_trainer.py)

### 繁體中文
我會提出一個拆分式架構：一個 edge service、兩個 online service、再加幾個非同步 worker。

```text
Client / Frontend
        |
        v
+----------------------+
| Gateway API          |
| auth / rate limit    |
| routing / upload     |
+----+-------------+---+
     |             |
     |             |
     v             v
+------------------+     +---------------------------+
| Recommendation   |     | Interaction Ingest        |
| Service          |     | Service                   |
+----+---------+---+     +-------------+-------------+
     |         |                       |
     |         v                       v
     |    +---------+            +-----------+
     |    | Redis   |<---------->| Kafka     |
     |    +----+----+            +-----+-----+
     |         |                       |
     v         v                       v
+----------------------+      +----------------------+
| RecommendationEngine |      | Content Worker       |
| RankingModel         |      | Feature Worker       |
| VectorSearch / TT    |      | Model Trainer        |
+----------------------+      +----------------------+
```

這也正好對應目前 codebase 的實作：

- [`gateway_api.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/gateway_api.py)
- [`recommendation_api.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/recommendation_api.py)
- [`interaction_ingest_api.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/interaction_ingest_api.py)
- [`kafka_workers/video_processor.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/kafka_workers/video_processor.py)
- [`kafka_workers/feature_updater.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/kafka_workers/feature_updater.py)
- [`model_trainer.py`](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/model_trainer.py)

---

## 9. Main Request Flows

### 9.1 Recommendation Serving Flow

#### English

1. Client sends `POST /api/recommendations` to the gateway.
2. Gateway handles API key validation and rate limiting.
3. Gateway forwards the request to `recommendation-service`.
4. Recommendation service checks Redis cache.
5. On cache miss, it loads user features and optional content features.
6. `RecommendationEngine` generates candidates from:
   - Two-Tower retrieval
   - content-based vector search
   - trending fallback
7. `RankingModel` reranks the merged candidate set.
8. The final ranked list is cached in Redis.
9. The service asynchronously emits a recommendation event to Kafka.
10. Response is returned to the client.

#### 繁體中文

1. Client 對 gateway 發 `POST /api/recommendations`。
2. Gateway 做 API key 驗證與 rate limiting。
3. Gateway 把請求轉發到 `recommendation-service`。
4. Recommendation service 先查 Redis cache。
5. 若 cache miss，就載入 user features 與可選的 content features。
6. `RecommendationEngine` 從以下來源產生 candidates：
   - Two-Tower retrieval
   - content-based vector search
   - trending fallback
7. `RankingModel` 對合併後候選集重排。
8. 最終結果寫回 Redis cache。
9. 服務再非同步送 recommendation event 到 Kafka。
10. 最後把結果回給 client。

### 9.2 Interaction Ingestion Flow

#### English

1. Client sends `POST /api/interactions`.
2. Gateway forwards the request to `interaction-ingest-service`.
3. The ingestion service tries Kafka first.
4. If Kafka is available, it returns `202 Accepted` immediately.
5. If Kafka is unavailable, it falls back to Redis Stream.
6. `feature-worker` consumes interaction events.
7. The worker batches updates and refreshes user features in Redis.

#### 繁體中文

1. Client 發送 `POST /api/interactions`。
2. Gateway 將請求轉發到 `interaction-ingest-service`。
3. Ingestion service 優先嘗試寫 Kafka。
4. 如果 Kafka 可用，立即回 `202 Accepted`。
5. 如果 Kafka 不可用，就回退到 Redis Stream。
6. `feature-worker` 消費互動事件。
7. Worker 做 batch 更新，並刷新 Redis 裡的 user features。

### 9.3 Content Upload Flow

#### English

1. Client uploads a video through `POST /api/content/upload`.
2. Gateway validates file type and priority.
3. Gateway stores the file temporarily.
4. Gateway writes `pending` status into Redis.
5. Gateway sends a task to Kafka topic `video-processing-tasks`.
6. `content-worker` processes the video asynchronously.
7. Extracted features are stored in Redis.
8. Visual embeddings are added into FAISS.
9. Content status is updated to `completed` or `failed`.

#### 繁體中文

1. Client 透過 `POST /api/content/upload` 上傳影片。
2. Gateway 驗證檔案格式與優先級。
3. Gateway 先把檔案暫存起來。
4. Gateway 在 Redis 中寫入 `pending` 狀態。
5. Gateway 將任務送到 Kafka topic `video-processing-tasks`。
6. `content-worker` 以非同步方式處理影片。
7. 抽出的內容特徵寫入 Redis。
8. 視覺 embedding 加入 FAISS。
9. 內容狀態更新成 `completed` 或 `failed`。

---

## 10. Storage Design

### English
I would explain storage by separating online state, event log, and retrieval index.

Redis stores:

- user features
- content features
- content status
- recommendation cache
- interaction logs
- analytics counters

Kafka stores the event stream:

- `user-interactions`
- `video-processing-tasks`
- `recommendation-events`
- `feature-updates`

FAISS stores vector indexes for low-latency ANN retrieval:

- content similarity index
- Two-Tower item embedding index

### 繁體中文
我會把 storage 拆成三層來講：online state、event log、retrieval index。

Redis 儲存：

- user features
- content features
- content status
- recommendation cache
- interaction logs
- analytics counters

Kafka 保存事件流：

- `user-interactions`
- `video-processing-tasks`
- `recommendation-events`
- `feature-updates`

FAISS 保存向量索引，支援低延遲 ANN 檢索：

- content similarity index
- Two-Tower item embedding index

---

## 11. Caching Strategy

### English
Caching is critical because recommendation serving is the latency-sensitive path.

I would cache:

- final recommendation results per user + context hash,
- user features,
- content features,
- trending items.

Why this helps:

- avoids repeated ranking work for identical contexts,
- reduces Redis and model computation load,
- lowers tail latency.

Risk:

- stale recommendations after new interactions.

Mitigation:

- short TTLs,
- adaptive TTL based on activity,
- invalidation after important interaction updates.

### 繁體中文
快取非常關鍵，因為推薦服務是最吃延遲的路徑。

我會快取：

- 依 user + context hash 的最終推薦結果，
- user features，
- content features，
- trending items。

好處：

- 避免同樣 context 重複做 ranking，
- 降低 Redis 與模型計算負載，
- 壓低 tail latency。

風險：

- 使用者剛互動完，推薦結果可能暫時過舊。

緩解方式：

- 使用較短 TTL，
- 根據活躍度做 adaptive TTL，
- 關鍵互動後觸發 cache invalidation。

---

## 12. Asynchronous Processing Design

### English
This is one of the main interview points. I would emphasize that writes and heavy processing should not block the online path.

Async tasks include:

- content processing
- feature updates
- recommendation event logging
- model retraining

Benefits:

- better latency isolation,
- better throughput,
- easier service specialization.

Trade-off:

- eventual consistency instead of instant consistency.

### 繁體中文
這會是面試中的重點之一。我會強調重寫入與重計算不應阻塞線上請求。

非同步任務包含：

- 內容處理
- 特徵更新
- 推薦事件記錄
- 模型重訓

優點：

- 延遲隔離更好，
- 吞吐更高，
- 服務分工更清楚。

代價：

- 系統會變成 eventual consistency，而不是即時強一致。

---

## 13. Scaling Strategy

### English
I would scale each subsystem independently.

Gateway:

- scale horizontally behind a load balancer,
- keep it stateless.

Recommendation service:

- scale by worker count and service replicas,
- keep model loading isolated from the gateway,
- optimize CPU and memory separately.

Interaction ingest service:

- scale for write-heavy traffic,
- rely on Kafka partitioning.

Workers:

- scale consumer count by topic throughput and partition count.

Redis:

- start with a single primary for simplicity,
- move to clustered or sharded design when memory or throughput becomes a bottleneck.

Kafka:

- increase partitions as ingestion grows.

### 繁體中文
我會讓每個子系統獨立擴容。

Gateway：

- 放在 load balancer 後面做 horizontal scaling，
- 保持 stateless。

Recommendation service：

- 透過 worker 數與 service replica 擴容，
- 模型載入與 gateway 隔離，
- 分開優化 CPU 與記憶體。

Interaction ingest service：

- 針對寫入流量做擴容，
- 依賴 Kafka partitioning。

Workers：

- 根據 topic 吞吐量與 partition 數增加 consumer 數量。

Redis：

- 一開始先用單主節點保持簡單，
- 當記憶體或吞吐成為瓶頸時再升級到 cluster 或 sharding。

Kafka：

- 隨著事件量成長增加 partitions。

---

## 14. Reliability and Failure Handling

### English
I would explicitly discuss how the system behaves under failure.

Examples:

- If Kafka is down, interaction ingestion falls back to Redis Stream.
- If recommendation logic fails, the service falls back to trending recommendations.
- If content processing fails, the content status becomes `failed` instead of blocking the user request.
- Health endpoints expose service and dependency status.
- Metrics allow alerting on latency, failure rate, and queue lag.

This is important in interviews because resilience is often more valuable than a perfect happy path.

### 繁體中文
我會明確說明系統在 failure 下怎麼運作。

例如：

- 如果 Kafka 壞掉，interaction ingestion 會回退到 Redis Stream。
- 如果推薦主邏輯失敗，服務回退到 trending recommendations。
- 如果內容處理失敗，content status 會變成 `failed`，而不是卡住使用者請求。
- Health endpoint 會暴露服務與依賴健康狀態。
- Metrics 可以用來做延遲、錯誤率、queue lag 告警。

這在面試裡很重要，因為有韌性的設計通常比只有 happy path 的設計更有價值。

---

## 15. Observability

### English
I would add observability from day one.

This codebase already exposes:

- `/health`
- `/metrics`
- structured request logging
- request IDs
- per-service runtime metrics
- recommendation profiling breakdowns

Operational metrics I would care about:

- recommendation latency P50/P95/P99
- cache hit rate
- Kafka consumer lag
- content processing success rate
- ranking time and candidate generation time
- Redis latency and error rate

### 繁體中文
我會從第一天就把 observability 放進系統。

這個 codebase 目前已經提供：

- `/health`
- `/metrics`
- structured request logging
- request IDs
- 每個服務的 runtime metrics
- recommendation profiling breakdown

我最關心的營運指標會是：

- recommendation latency P50/P95/P99
- cache hit rate
- Kafka consumer lag
- content processing success rate
- ranking time 與 candidate generation time
- Redis latency 與 error rate

---

## 16. Deep Dive on Recommendation Logic

### English
In interviews, one strong move is to explain why a single retrieval strategy is not enough.

This design uses multiple candidate sources:

- collaborative filtering via Two-Tower retrieval
- content-based retrieval via vector similarity
- trending/popularity retrieval

Then it applies a ranking model.

Why this is good:

- collaborative retrieval captures long-term user taste,
- content retrieval captures short-term intent from the current video,
- trending protects the system from sparse data and cold start,
- ranking combines multiple signals into one final decision.

### 繁體中文
在面試裡，一個很加分的點是說清楚為什麼單一召回策略不夠。

這個設計用了多種 candidate source：

- 透過 Two-Tower 做 collaborative filtering retrieval
- 透過 vector similarity 做 content-based retrieval
- 透過 trending/popularity 做兜底召回

最後再交給 ranking model 做排序。

這樣的好處是：

- collaborative retrieval 能抓長期偏好，
- content retrieval 能抓目前影片帶來的短期意圖，
- trending 能處理稀疏資料與 cold start，
- ranking 把多種訊號合併成最後決策。

---

## 16A. Serving State in Redis

### English
If I want this document to reflect the current codebase more precisely, I should describe Redis as more than a feature store.

In the current implementation, Redis is also part of the serving-control plane and stores:

- recommendation cache
- per-user candidate cache
- segment candidate cache
- versioned user embedding cache
- trending serving pools
- category serving pools
- CF model version metadata
- product metadata cache

This matters because the recommendation path is no longer purely “live retrieve and rank.” A meaningful portion of serving state is precomputed or cached in Redis so the online path can avoid recomputing the same work repeatedly.

### 繁體中文
如果要讓這份文件更貼近目前的 codebase，就不能只把 Redis 當成 feature store。

在目前實作裡，Redis 也同時是 serving-control plane 的一部分，會存放：

- recommendation cache
- per-user candidate cache
- segment candidate cache
- versioned user embedding cache
- trending serving pools
- category serving pools
- CF model version metadata
- product metadata cache

這很重要，因為現在的推薦路徑已經不是單純「即時召回再排序」。有一部分 serving state 會先在 Redis 中預先準備或快取，讓線上請求不需要重複做相同計算。

---

## 16B. Cache Hierarchy and Precomputed Pools

### English
The current serving path uses a cache hierarchy instead of a single cache layer.

Conceptually, the recommendation service checks these layers in order:

1. final recommendation cache
2. per-user candidate cache
3. segment candidate cache
4. live candidate generation

On top of that, the system precomputes reusable serving pools such as:

- global trending pools
- category-specific pools
- optional hot-segment candidate pools based on page, device, category, and recent content

This is a classic latency-throughput trade-off. We spend more memory and background compute so the recommendation service can avoid doing full candidate generation for every request. It also gives us better resilience during traffic spikes because hot cohorts can share cached candidate sets.

### 繁體中文
目前的 serving path 採用的是多層 cache hierarchy，而不是只有一層快取。

概念上，recommendation service 會依序檢查：

1. final recommendation cache
2. per-user candidate cache
3. segment candidate cache
4. live candidate generation

除此之外，系統還會預先計算可重複使用的 serving pools，例如：

- global trending pools
- category-specific pools
- 依 page、device、category、recent content 組成的 hot-segment candidate pools

這是一個很典型的 latency 與 throughput trade-off。我們用更多記憶體與背景計算，換取 recommendation service 不需要對每個 request 都完整跑一次 candidate generation。這也能讓系統在流量尖峰時更穩，因為熱門 cohort 可以共享同一批候選集。

---

## 16C. Versioned Two-Tower Embeddings and Model Coordination

### English
The current recommendation stack has also evolved from “train a Two-Tower model and load it” into a more coordinated versioned serving flow.

The important idea is that the trainer, serving layer, and feature updater must agree on the active collaborative-filtering model version.

That coordination enables:

- loading versioned Two-Tower checkpoints
- loading the matching CF ANN index
- publishing user embeddings tied to a specific model version
- resolving cached user embeddings safely during serving
- invalidating or refreshing embeddings when the model version changes

This design reduces unnecessary user-embedding recomputation on the hot path. It also avoids a subtle correctness problem where a user embedding generated by one model version might be used against an item index generated by another version.

### 繁體中文
目前的推薦堆疊也已經從「訓練一個 Two-Tower model 然後載入」演進成更完整的版本化 serving 流程。

關鍵概念是：trainer、serving layer 與 feature updater 必須對目前啟用的 collaborative-filtering model version 有一致認知。

這種協調可以支援：

- 載入 versioned Two-Tower checkpoints
- 載入對應版本的 CF ANN index
- 發布綁定特定 model version 的 user embeddings
- serving 時安全地讀取快取中的 user embeddings
- 在 model version 變更時做 invalidation 或 refresh

這樣做可以減少 hot path 上不必要的 user embedding 重算，也能避免一個隱性正確性問題：也就是用舊版模型產生的 user embedding，去查新版 item index。

---

## 16D. Ranking Batching and Cheap Pre-rank

### English
The current codebase also includes ranking-stage optimizations that are worth mentioning in an interview because they show practical systems thinking.

Instead of assuming every request calls the ranker independently, the system supports:

- async ranking micro-batching
- configurable batch size and wait time
- a cheap pre-rank phase that trims the candidate set before full ranking

Why this helps:

- micro-batching improves CPU efficiency under concurrency,
- a small batch wait can improve throughput without materially hurting latency,
- cheap pre-rank reduces the cost of full neural ranking by only sending the top M candidates forward.

The trade-off is that batching adds queueing complexity and a little extra tail-latency risk. So this optimization is best when traffic is high enough that batching pays for itself.

### 繁體中文
目前的 codebase 也已經有 ranking 階段的優化，這在面試中很值得提，因為它代表的是偏實務的系統思維。

系統不再假設每個 request 都單獨直接呼叫 ranker，而是支援：

- async ranking micro-batching
- 可設定的 batch size 與 wait time
- cheap pre-rank 階段，先把 candidate set 縮小再做完整 ranking

這些優化的好處是：

- micro-batching 能在高併發下提升 CPU 使用效率，
- 短暫 batch wait 能提升吞吐量，而不一定明顯增加延遲，
- cheap pre-rank 會先裁掉不夠好的 candidates，降低完整 neural ranking 的成本。

代價是 batching 會增加 queueing complexity，也會帶來一些 tail latency 風險。所以這種優化最適合在流量夠高、batching 能真正回本的情境。

---

## 16E. Current Deployment and Observability Reality

### English
If I am describing the current repository rather than an abstract future design, I should also be explicit about the present deployment and observability reality.

Current deployment shape:

- one gateway service
- one recommendation service
- one interaction-ingest service
- one content worker
- one feature worker
- one model-trainer service
- one Redis instance
- one Kafka broker plus ZooKeeper
- one `kafka-init` bootstrap job for topic creation
- Prometheus and Grafana

This is production-shaped, but it is not yet a fully HA multi-node deployment. It is better described as a single-cluster, single-broker deployment that is good for development, benchmarking, and early production-style iteration.

Current observability shape:

- request IDs propagated through middleware
- structured JSON logging
- per-route HTTP counters and latency histograms
- in-progress request gauges
- exception counters
- process CPU and resident memory gauges
- Redis runtime metrics
- Kafka producer health and consumer lag metrics

This matters in interviews because a strong design is not only about data flow. It also needs operational visibility: we should be able to answer whether latency is increasing, whether Kafka lag is growing, and whether Redis or ranking is becoming the bottleneck.

### 繁體中文
如果我描述的是目前這個 repository，而不是抽象未來架構，那我也應該明確說出現在的 deployment 與 observability 現況。

目前部署形態是：

- 一個 gateway service
- 一個 recommendation service
- 一個 interaction-ingest service
- 一個 content worker
- 一個 feature worker
- 一個 model-trainer service
- 一個 Redis instance
- 一個 Kafka broker 加上 ZooKeeper
- 一個用來建立 topics 的 `kafka-init` bootstrap job
- Prometheus 與 Grafana

這個拓撲已經有 production 的輪廓，但還不是完整 HA 的多節點部署。更精確地說，它是單叢集、單 broker 的部署，適合開發、壓測與 early production-style iteration。

目前 observability 形態包含：

- 透過 middleware 傳遞 request IDs
- structured JSON logging
- per-route HTTP counters 與 latency histograms
- in-progress request gauges
- exception counters
- process CPU 與 resident memory gauges
- Redis runtime metrics
- Kafka producer health 與 consumer lag metrics

這在面試裡很重要，因為好的設計不只是在講 data flow，也要能回答營運問題：例如 latency 是否正在升高、Kafka lag 是否正在累積、Redis 或 ranking 是否正在成為瓶頸。

---

## 17. Trade-Off Discussion

### English
This is the part interviewers usually care about most. I would make the trade-offs explicit.

#### 17.1 Microservices vs Monolith

Microservices advantages:

- independent scaling,
- clearer ownership,
- latency-heavy and write-heavy workloads can be isolated.

Microservices disadvantages:

- more operational complexity,
- more network hops,
- harder local debugging.

Why I still choose microservices here:

- recommendation serving and interaction ingestion have very different scaling patterns,
- content processing is naturally asynchronous and worker-based,
- this repository already reflects those boundaries.

#### 17.2 Redis vs SQL Database for Online Features

Redis advantages:

- low latency,
- simple TTL support,
- good fit for cache and session-like feature state.

Redis disadvantages:

- weaker query model,
- memory cost is higher,
- durability guarantees are weaker than a primary source of truth database.

Why Redis is acceptable:

- online serving values latency over complex querying,
- the feature store is not the only long-term system of record.

#### 17.3 Kafka vs Direct Synchronous Updates

Kafka advantages:

- decouples services,
- smooths traffic spikes,
- enables replay and multiple consumers.

Kafka disadvantages:

- eventual consistency,
- queue lag,
- more infra to operate.

Why Kafka is a good fit:

- interactions and video processing do not need to complete synchronously in the user request path.

#### 17.4 ANN Search vs Exact Search

ANN advantages:

- much lower latency at scale,
- practical for online vector retrieval.

ANN disadvantages:

- approximate results,
- tuning complexity.

Why ANN is the right choice:

- recommendation systems care about response time and good-enough recall, not exact nearest neighbors every time.

#### 17.5 Precompute vs Real-Time Ranking

Precompute advantages:

- cheaper per request,
- stable latency.

Precompute disadvantages:

- stale results,
- weak personalization for rapidly changing context.

Real-time ranking advantages:

- fresher and more personalized.

Real-time ranking disadvantages:

- higher compute cost.

Why hybrid wins:

- precompute some state,
- retrieve online,
- rank online for the final top K.

### 繁體中文
這通常是面試官最在意的部分，所以我會把 trade-off 明確講出來。

#### 17.1 Microservices vs Monolith

Microservices 優點：

- 可獨立擴容，
- ownership 更清楚，
- 高延遲與高寫入工作負載可以隔離。

Microservices 缺點：

- 營運複雜度更高，
- network hop 更多，
- 本地除錯更難。

為什麼這裡仍選 microservices：

- recommendation serving 和 interaction ingestion 的擴展模式差很多，
- content processing 天生就是非同步 worker 型工作，
- 這個 repo 目前也已經自然長成這些邊界。

#### 17.2 Redis vs SQL Database for Online Features

Redis 優點：

- 延遲低，
- TTL 支援簡單，
- 很適合 cache 與 session-like feature state。

Redis 缺點：

- 查詢能力較弱，
- 記憶體成本較高，
- durability 不如真正的 system of record。

為什麼 Redis 可以接受：

- 線上 serving 更重視低延遲，而不是複雜查詢，
- feature store 也不應該是唯一長期資料來源。

#### 17.3 Kafka vs Direct Synchronous Updates

Kafka 優點：

- 服務解耦，
- 能吸收流量尖峰，
- 支援 replay 與多 consumer。

Kafka 缺點：

- 變成 eventual consistency，
- 會有 queue lag，
- 基礎設施更複雜。

為什麼 Kafka 適合：

- interaction 與 video processing 不需要在使用者 request path 內同步完成。

#### 17.4 ANN Search vs Exact Search

ANN 優點：

- 在規模上延遲低很多，
- 很適合線上向量檢索。

ANN 缺點：

- 結果是 approximate，
- tuning 較複雜。

為什麼這裡選 ANN：

- 推薦系統要的是快且夠好的召回，不是每次都求精確最近鄰。

#### 17.5 Precompute vs Real-Time Ranking

Precompute 優點：

- 每次請求成本較低，
- latency 穩定。

Precompute 缺點：

- 結果容易過時，
- 面對快速變化的 context 時個人化較弱。

Real-time ranking 優點：

- 更新鮮、個人化更強。

Real-time ranking 缺點：

- 計算成本更高。

為什麼 hybrid 最合理：

- 一部分 state 預先準備，
- 召回在線上做，
- 最後 top K 再在線上排序。

---

## 18. What I Would Build in V1 vs Later

### English
If the interviewer asks for prioritization, I would say:

V1:

- gateway
- recommendation service
- interaction ingestion
- Redis feature store
- Kafka event backbone
- one vector index
- simple ranking model
- observability

Later:

- richer content understanding
- better model retraining pipeline
- A/B testing framework
- feature store backed by offline warehouse + online sync
- stronger multi-region story
- more advanced exploration/diversity controls

### 繁體中文
如果面試官問優先順序，我會這樣回答：

V1：

- gateway
- recommendation service
- interaction ingestion
- Redis feature store
- Kafka event backbone
- 一套向量索引
- 簡單 ranking model
- observability

之後再做：

- 更完整的內容理解能力
- 更成熟的模型重訓 pipeline
- A/B testing framework
- 由 offline warehouse 支撐、同步到 online 的 feature store
- 更完整的多區部署能力
- 更進階的 exploration / diversity 控制

---

## 19. Final Interview Wrap-Up

### English
If I had 30 seconds to summarize, I would say:

This is a low-latency recommendation system for video commerce. I split the architecture into a gateway, an online recommendation service, an interaction ingestion service, and asynchronous workers. Redis handles online features and caching, Kafka decouples ingestion and processing, and FAISS powers vector retrieval. The system uses hybrid retrieval plus ranking, scales each component independently, and accepts eventual consistency where it improves latency and throughput. The main trade-off is added operational complexity in exchange for better isolation, scalability, and recommendation freshness.

### 繁體中文
如果最後只剩 30 秒要收尾，我會這樣講：

這是一個面向短影音電商的低延遲推薦系統。我把架構拆成 gateway、線上 recommendation service、interaction ingestion service，以及非同步 workers。Redis 負責線上特徵與快取，Kafka 負責事件解耦，FAISS 負責向量召回。整體採用 hybrid retrieval 加 ranking，各元件可獨立擴容，並接受 eventual consistency 來換取更好的延遲與吞吐。最大的 trade-off 是多了一些系統複雜度，但換來更好的隔離性、擴展性與推薦新鮮度。

---

## 20. Interview Speaking Script

### English
This section is a more conversational version that I could actually speak during an interview. Each block is designed to fit in roughly 1 to 2 minutes.

### 繁體中文
這一節是更口語化的版本，適合在面試中直接講。每一段都控制在大約 1 到 2 分鐘內可以說完。

### 20.1 Opening and Scope

#### English
Let me first clarify the scope. I am designing a recommendation system for a short-video commerce platform. The core problem is that users watch product-related videos, interact with items, and expect low-latency personalized recommendations. So the system needs to handle three things well: first, online recommendation serving; second, real-time ingestion of user interactions; and third, asynchronous processing of uploaded video content into searchable features. For a first version, I would keep payments, merchant workflow, and global multi-region replication out of scope so I can focus on recommendation quality, latency, and system reliability.

#### 繁體中文
我先釐清這題的 scope。我要設計的是一個短影音電商平台的推薦系統。核心問題是使用者會看商品相關影片、和商品互動，並且期待系統可以低延遲地回傳個人化推薦。所以系統要先把三件事做好：第一，線上推薦服務；第二，即時接收使用者互動事件；第三，把上傳的影片非同步處理成可搜尋的內容特徵。至於第一版，我會先把支付、商家後台流程、全球多區複寫放在 scope 外，先專注在推薦品質、延遲與可靠性。

### 20.2 Requirements and Scale

#### English
For requirements, functionally the system should accept recommendation requests, interaction events, and content uploads. It should support multiple recommendation strategies, including collaborative filtering, content-based retrieval, and trending fallback. Non-functionally, my main goals are low latency on the recommendation path, high write throughput on the interaction path, and graceful degradation when one dependency is unhealthy. For rough scale, I would assume millions of daily active users, peak recommendation traffic around one thousand requests per second, and interaction traffic much higher, maybe tens of thousands of events per second. That immediately tells me I should not keep everything inside one synchronous service.

#### 繁體中文
需求上，功能面系統要能接收推薦請求、互動事件，以及內容上傳。推薦策略上至少要支援 collaborative filtering、content-based retrieval，還有 trending fallback。非功能面，我最重視的是推薦路徑要低延遲、互動路徑要高吞吐，而且當某個依賴出問題時，系統要能 graceful degradation。流量估算上，我會假設有數百萬到上千萬日活，推薦流量尖峰大概每秒一千次，而互動事件流量會更高，可能是每秒數萬筆。這個量級馬上告訴我，不能把所有事情都放在單一同步服務裡面做。

### 20.3 High-Level Architecture

#### English
At a high level, I would split the system into a gateway, a recommendation service, an interaction ingestion service, and a set of asynchronous workers. The gateway is the single external entry point and handles authentication, rate limiting, request validation, request IDs, and routing. The recommendation service is optimized for low-latency online inference. The interaction ingestion service is optimized for high-throughput writes and returns quickly after accepting events. Then I use background workers for video processing, feature updates, and model retraining. Redis is not only a feature store here; it also holds serving state such as recommendation caches, candidate caches, segment caches, user embeddings, and precomputed pools. Kafka is the event backbone, and FAISS provides low-latency vector retrieval. This separation lets me scale serving and ingestion independently while still sharing state through Redis and Kafka.

#### 繁體中文
在 high-level architecture 上，我會把系統拆成 gateway、recommendation service、interaction ingestion service，以及一組非同步 workers。Gateway 是唯一對外入口，負責 authentication、rate limiting、request validation、request IDs 和 routing。Recommendation service 專門優化低延遲的線上推論。Interaction ingestion service 則專門優化高吞吐寫入，接到事件後盡快回應。接著再用背景 workers 處理影片內容、更新特徵，以及做模型重訓。這裡的 Redis 不只是 feature store，也會保存 serving state，像是 recommendation caches、candidate caches、segment caches、user embeddings 與預先計算好的 pools。Kafka 當事件骨幹，FAISS 則做低延遲向量召回。這樣拆開後，推薦與寫入就可以獨立擴容，同時又能透過 Redis 與 Kafka 共享狀態。

### 20.4 Recommendation Flow

#### English
Now I will walk through the recommendation flow. The client sends a recommendation request to the gateway with a user ID, optional content ID, and some context. The gateway validates the request and forwards it to the recommendation service. The recommendation service first checks a cache hierarchy because that is the latency-sensitive path. It can hit a final recommendation cache, then a candidate cache, then a segment-level candidate cache for hot cohorts, before falling back to full live retrieval. On a live path, it loads user features and, if present, content features for the current video. Then the recommendation engine generates candidates from multiple sources: collaborative retrieval using a Two-Tower model, content-based retrieval from vector search, and trending or category pools as fallback. The collaborative side can also reuse a prepublished user embedding tied to the current model version. After candidate generation, the system can do a cheap pre-rank and then send the remaining set through batched ranking. The final top K is cached, and recommendation events are logged asynchronously for analytics and future training.

#### 繁體中文
接著我走一遍推薦流程。Client 會把 user ID、可選的 content ID，還有一些 context 傳到 gateway。Gateway 做完驗證後，把請求轉發到 recommendation service。Recommendation service 會先檢查一個多層 cache hierarchy，因為這是最吃延遲的路徑。它可能先命中 final recommendation cache，再來是 candidate cache，再來是針對熱門 cohort 的 segment-level candidate cache，最後才回到完整的 live retrieval。走 live path 時，它會讀 user features；如果有 content ID，也會把目前影片的 content features 一起拿進來。接著 recommendation engine 會從多個來源做召回：包含 Two-Tower 的 collaborative retrieval、vector search 的 content-based retrieval，以及 trending 或 category pools 作為 fallback。Collaborative 這一側也可以直接重用和目前 model version 對齊的 prepublished user embedding。候選集產生後，系統可以先做 cheap pre-rank，再把剩下的集合送進 batched ranking。最後 top K 會寫回 cache，並且非同步記錄 recommendation event，供分析和後續訓練使用。

### 20.5 Interaction and Content Processing Flow

#### English
For the write-heavy path, I want the API to return quickly. So when the client sends an interaction event, the gateway forwards it to the interaction ingestion service. That service tries to push the event into Kafka first and immediately returns a 202 if successful. If Kafka is unavailable, I can degrade to Redis Stream so the system still accepts events. A feature worker later consumes those events in batches, updates user features, invalidates stale serving caches, and can publish a refreshed user embedding for the current collaborative model version. For content uploads, the flow is similar. The gateway stores the uploaded video temporarily, writes a pending status, and sends a task to Kafka directly. A content worker processes the video asynchronously, extracts embeddings and metadata, stores content features, updates the FAISS index, and marks the content as completed or failed.

#### 繁體中文
在高寫入路徑上，我希望 API 能很快回應。所以當 client 上報互動事件時，gateway 會把請求轉到 interaction ingestion service。這個服務會優先把事件送進 Kafka，如果成功就立刻回 `202 Accepted`。如果 Kafka 暫時不可用，就回退到 Redis Stream，確保系統還是能接收事件。後面再由 feature worker 批次消費這些事件，更新 user features、清掉已過時的 serving caches，並且在需要時重新發布和目前 collaborative model version 對齊的 user embedding。至於內容上傳，流程也很像。Gateway 先把影片暫存起來，寫入 pending 狀態，然後直接把任務送到 Kafka。接著 content worker 非同步處理影片，抽出 embedding 與 metadata，存回內容特徵、更新 FAISS index，最後把狀態標成 completed 或 failed。

### 20.6 Storage, Caching, and Consistency

#### English
For storage, I would separate online state, streaming data, and vector retrieval. Redis is my online store for both features and serving state, so it keeps user features, content features, content status, final recommendation caches, candidate caches, segment caches, product metadata, and versioned user embeddings. Kafka is the durable event stream that decouples producers from consumers. FAISS stores the vector indexes for content similarity and collaborative item retrieval. On caching, I am intentionally using a hybrid strategy: some state is precomputed, some is cached on demand, and the final top K is still produced online. The trade-off is staleness versus latency. I accept eventual consistency here because fresh interaction updates do not need to synchronously block recommendation serving. I mitigate that with short TTLs, adaptive TTLs, cache invalidation, and version-aware embedding refresh.

#### 繁體中文
在 storage 設計上，我會把 online state、streaming data 與 vector retrieval 分開來看。Redis 是線上資料層，同時也保存 features 和 serving state，所以裡面會有 user features、content features、content status、final recommendation caches、candidate caches、segment caches、product metadata，以及 versioned user embeddings。Kafka 是耐久的事件流，用來把 producer 和 consumer 解耦。FAISS 則保存內容相似度與 collaborative item retrieval 所需的向量索引。快取方面，我會刻意採用 hybrid strategy：有些 state 預先計算，有些在需要時快取，而最後的 top K 仍然在線上產生。這裡的 trade-off 是 staleness 和 latency 之間的交換。我願意接受 eventual consistency，因為互動更新不需要同步阻塞推薦服務。我會用短 TTL、adaptive TTL、cache invalidation，以及 version-aware embedding refresh 來降低過時問題。

### 20.7 Scaling, Reliability, and Trade-Offs

#### English
For scaling, I would scale the gateway horizontally as a stateless service and tune its internal proxy connection pool because it is the single public entry point. I would scale recommendation service replicas separately from ingestion replicas because their resource profiles are different. Kafka partitions let me scale consumers, and workers can be added independently depending on content volume or event lag. Inside recommendation serving, I would also use precomputed pools, cache hierarchy, and ranking micro-batching to improve concurrency efficiency. For reliability, I want clear fallback behavior: if Kafka is unhealthy, ingestion falls back; if recommendation logic fails, I can return trending recommendations; if content processing fails, I mark the content as failed instead of blocking the user request. Operationally, I would monitor request latency, cache hit rate, Redis health, Kafka consumer lag, process CPU and memory, and ranking-stage timing. The main trade-off in this design is choosing a more complex microservice architecture with precompute layers over a simpler monolith. I accept that complexity because the workloads are very different, and the separation gives me better scalability, fault isolation, and latency control.

#### 繁體中文
在擴展性方面，我會把 gateway 當成 stateless service 來做 horizontal scaling，並且調整它的內部 proxy connection pool，因為它是唯一公開入口。Recommendation service 和 ingestion service 會分開擴容，因為兩者的資源型態完全不同。Kafka partitions 可以讓我擴 consumer，workers 也能依照內容量或 event lag 獨立增加。在 recommendation serving 內部，我也會利用 precomputed pools、cache hierarchy，以及 ranking micro-batching 來提高高併發下的效率。可靠性上，我會設計清楚的 fallback 行為：如果 Kafka 不健康，ingestion 會回退；如果推薦主邏輯失敗，可以先回 trending recommendations；如果內容處理失敗，就把內容標成 failed，而不是卡住使用者請求。在營運層面，我會監控 request latency、cache hit rate、Redis health、Kafka consumer lag、process CPU 和 memory，以及 ranking 階段的耗時。這個設計最大的 trade-off 是我選擇了一個比 monolith 更複雜、而且帶有 precompute layers 的 microservice 架構。我接受這個複雜度，因為各種工作負載差異很大，而拆分能帶來更好的可擴展性、故障隔離與延遲控制。

### 20.8 Closing Summary

#### English
So to summarize, I would design this as a hybrid recommendation platform with online serving, asynchronous ingestion, and asynchronous content processing. Redis is both the online feature layer and a serving-state layer, Kafka is the decoupling and buffering layer, and FAISS is the retrieval layer. Recommendation quality comes from combining collaborative, content-based, and trending signals, while performance comes from cache hierarchy, precomputed pools, versioned embeddings, and batched ranking. The core trade-off is accepting eventual consistency and more operational complexity in exchange for lower latency, better throughput, and fresher recommendations.

#### 繁體中文
總結來說，我會把它設計成一個 hybrid recommendation platform，包含線上 serving、非同步 ingestion，以及非同步內容處理。Redis 不只是線上特徵層，也是 serving-state layer；Kafka 是解耦與緩衝層；FAISS 是召回層。推薦品質來自 collaborative、content-based 與 trending 訊號的結合，而效能則來自 cache hierarchy、precomputed pools、versioned embeddings 與 batched ranking。核心 trade-off 是：我接受 eventual consistency 和更高的營運複雜度，來換取更低延遲、更高吞吐，以及更新鮮的推薦結果。
