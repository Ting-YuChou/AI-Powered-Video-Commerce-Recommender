# 專案擴展與強化檢視

- 首版：2026-10-07，基準 `main` `5695367`。
- 修訂：2026-10-20，基準 `codex/ranking-phase-c` `f95f378`（內容與 `origin/main` `1e860c2` 相同）。文中 `file:line` 皆指修訂基準。

本次修訂重新掃描整個 split-service 專案，分六個面向進行：線上資料管線、離線／PIT 訓練資料、排序模型、召回與多模態、線上服務 runtime、部署／CI／監控／前端。每個面向都逐項核對首版 23 項在目前程式碼的狀態，並補上新發現。所有 P0 與多數關鍵 P1 已直接讀原始碼確認，或用小型 probe 重現；其餘項目以程式路徑追蹤為主，沒有在真實 Kafka／Flink／K8s 環境驗證（見第五節）。本報告不改 runtime、模型或部署設定；優先級代表工作排序，效益描述是預期方向，不是已量測的改善。

## 修訂摘要

首版之後的 18 個 commit 已把首版 A／B 階段的大部分骨架做出來：

- 資料與事件：Flink-aware 啟動與 job 冪等提交；interaction idempotency ledger；served／viewed 曝光分離與 durable outbox。
- 內容處理：content-worker 例外重拋與 processing lease；上傳 outbox 與 content-task-publisher。
- 評估與上線：`business-value-v1` 共用 scoring policy；model release lifecycle（registered → validated → staging → active，加上 CAS）；retrieval PIT 與 Two-Tower release gate。
- 訓練與 artifact：PIT streaming reader 與 bounded-memory 訓練；原子 artifact generation；Iceberg maintenance。
- Phase C：categorical 特徵、RRF fusion、temperature calibration。
- CI：加入前端測試、Flink Maven、Helm／kubeconform／promtool 與 Flink closed-loop。

不過首版 23 項沒有一項能完整結案：17 項部分解決，6 項未處理（14、15、16、20、22、23）。新掃描另有 54 項發現（24–77），其中 P0 2 項、P1 29 項、P2 17 項、P3 6 項。

主要判斷更新：目前最大的風險不再是「缺少閉環」，而是**閉環本身的量測與預設路徑不可信**。

1. **新建的 gate 有量測錯誤。** Two-Tower gate 把 FAISS L2 距離當成相似度（24）；排序 NDCG 被全 view 的 slate 稀釋（45）；只要有一個長尾切片，gate 就永遠 INSUFFICIENT（46）；trainer 的 champion 不是現役模型（38）；PIT parity／leakage gauge 是寫死的值（41）。這些修好之前，gate 的 PASS／FAIL 不能當作模型決策依據。
2. **預設操作路徑會讓已修好的鏈路停擺。**
   - `./startup.sh restart` 不帶 savepoint 重新提交 Flink job，Redis 寫入先全部被判 stale，接著 conflict 讓 job 進入 crash loop（25）。
   - 單一超長欄位就讓 JDBC sink 無限重試（26）。
   - PIT preset 預設開啟 trimodal shadow，每輪都 fail-closed（36）。
   - Helm 的 content-worker image 沒有 PaddleOCR（68）。
   - 前端 Live 模式上傳送出的是 JSON（67）。
3. **edge 的可用性與安全契約不足。** readiness 級聯（59）、多 worker 下指標失真（61）、Postgres 熱路徑沒有 timeout（62）、Alertmanager 沒有 receiver（69）、Compose 預設對外開放且免驗證（70）。

建議先修「量測正確性」和「預設路徑可運作」（多數為 S 規模），再處理模型改善與擴容。模型改善項目（11、12、13、15、17、47、48）仍有價值，但要等 gate 可信後，才判斷得出有沒有效。

## Phase 0 實作進度（2026-10-20，未 commit）

| 項目 | 狀態 | 做了什麼 |
| --- | --- | --- |
| 24 | 程式已修 | Two-Tower 與 content／CF index 改用 IP；`score_ann_index` 依 `metric_type` 轉換分數，舊的 L2 champion 也能評估正確；bundle meta 升為 `two_tower_index_v2` 並記錄 metric；沒有標示 metric 的舊 content generation 會被拒絕並重建 |
| 45 | 程式已修 | gate NDCG 的 gain 改為 `relevance − VIEW_RELEVANCE`，只含 view 的 slate 不計入，另回報 `ndcg_informative_impressions`；policy 升為 `ranking_quality_gate_v3`。訓練 label 與 `ranking.py` 的 sklearn NDCG 沒動（Stage C） |
| 25 | 程式已修 | submitter 依序從 savepoint、最新 retained checkpoint、空 state 還原；sink 以 `flink:feature_state_owner` 拒絕「同 epoch、空 state、不同 job」的啟動；conflict 改為計數並略過；restart 上限為 10 分鐘內 10 次；新增 stale-writes 告警；`verify` 不再把 RESTARTING 視為健康；drill 加入 compose down/up 情境 |
| 26 | 程式已修 | Flink parser 依欄寬檢查，超長送 DLQ；views sink 改為 `ON CONFLICT DO NOTHING`；API 的 user/product/content ID 上限為 255；views 拒絕重複的 event_id；`X-Request-ID` 只接受 1–64 個安全字元 |
| 04 | 程式已修 | `payload_hash` 只在 client 實際帶了 event_time／timestamp 時才納入 |
| 36 | 程式已修 | Compose 與 Helm 的 PIT preset 都設 `RANKING_TRIMODAL_SHADOW=false`；沒有 PIT URI 時 shadow 會跳過並記錄 |
| 67 | 程式已修 | 移除 axios instance 的 JSON 預設 header；補 adapter 測試 |
| 68 | 程式已修 | 新增 `images.contentWorker` 與 workload `imageRef`；configmap 明確設定 `MODEL_OCR_BACKEND`；CI build content-worker target |
| 70 | 程式已修 | Caddy 預設綁 `EDGE_BIND_ADDRESS=127.0.0.1`；gateway 沒有 credential 時回 503 `AUTH_NOT_CONFIGURED`；本機預設為 `SECURITY_AUTH_MODE=disabled`。UI 不帶 key 仍是項目 22 |

**Docker 驗證（OrbStack）。**
- 通過的項目：
  - backend-tests：838 passed。
  - Maven：42 項。
  - promtool。
  - helm lint/template 加上 kubeconform：75 個資源。
  - 四個 image 都能 build；content-worker 內含 `/opt/paddleocr`。
  - Flink closed-loop smoke。
  - Flink recovery drill：新增的 compose down/up 情境中，job 從 checkpoint 還原，generation 由 4 接續到 5，並維持 RUNNING。
- 驗證過程中另外修了一個既有問題：zookeeper 沒有持久化，每次 down/up 之後 Kafka 都會出現 cluster ID 不一致。
- ranking-service 不依賴 coordinator 是刻意的設計：Triton profile 需要繞過 coordinator，有 contract test 鎖定。所以只從 ranking-service 帶起的 stack，要另外明確啟動 coordinator。

**尚未達成的完成條件。**
- 重新產生 retrieval／NDCG baseline。
- 在 K8s 上實際處理影片。
- UI Live 上傳的 e2e。
- pit-production 跑完一輪訓練。

手動的 `docker-integration` job 在全新 stack 上仍然無法通過，原因有四：沒有啟動 ranking coordinator；需要先跑 sample vector bootstrap；ranker 未訓練時處於 degraded，gateway 的 readiness 會連帶 not_ready（項目 59）；多個 runner replica 同時 `create_all` 會 race。這四點都不是 Phase 0 的修改造成的。

`ranking_quality_gate_v2` 的 evidence 目前仍被視為有效的 Phase C 晉升依據，所以既有 release 不受影響；是否強制改以 v3 重新評估，需要另外決定。

## 嚴重度與規模

- **P0**：預設路徑會靜默做出錯誤決策，或在例行操作下讓核心鏈路停擺。應立即處理。
- **P1**：在常見情境下，造成資料完整性、可用性、安全或模型評估失真。
- **P2**：擴展、正式上線前，或資料量成長後必須處理。
- **P3**：衛生與小幅改善。
- **規模**：S＝局部契約／設定，或數個小 PR；M＝跨模組實作與整合驗證；L＝橫跨資料／模型／部署路徑，通常需要多輪實驗。這是相對工程規模，不是工期承諾。

## 目前實際具備的能力

| 層級 | 程式碼現況 | 啟用／證據邊界 |
| --- | --- | --- |
| 線上服務 | Caddy → gateway → recommendation → ranking；預設走 coordinator/runner，Triton 為選配 | ASGI、bounded queue、deadline、429、artifact generation 原子切換都已實作。遠端排序拓撲下 retrieval artifact 不會熱切換（58）。不能據此宣稱整條鏈路達到指定 QPS |
| 召回 | Two-Tower（PIT 訓練＋release gate）、視覺向量、熱門／類別／cluster／新商品 pools；SASRec、Swing 為選配；legacy 合併或 shadow RRF | 預設是 legacy fusion。SASRec／Swing／trending 在 production 沒有產生路徑（51）。gate 的排序方向錯誤（24） |
| 排序 | DCN／low-rank DCNv2 多任務＋LTR；預設為 v1 28 維，v3 categorical 為 challenger；DIN、trimodal 為選配 | 已共用 business-value-v1 scoring。calibration 預設為 none。DIN 經 PIT 拿不到資料（34） |
| 多模態 | FFmpeg scene-adaptive frames、CLIP、PaddleOCR、E5；ASR 為選配 | OCR 預設英文（56）。主模態失敗時會靜默產生零向量（54）。trimodal 尚未進 ONNX |
| 即時資料 | Kafka → Flink（以 Lua 原子寫入 Redis）→ Redis／Postgres／feature history；idempotency ledger；served／viewed 曝光 | Compose 預設已含 Flink job。重新提交時不會 restore（25）。Helm 不部署 Flink（01） |
| 離線資料 | Kafka → Flink → Iceberg／S3 → PIT（ranking＋retrieval）→ streaming reader → trainer | 基礎設定仍是 legacy，PIT 要靠 preset 啟用。preset 預設的 trimodal shadow 會失敗（36）。parity gauge 是寫死的值（41） |
| 上線管理 | registered → validated → staging → active＋CAS；generation bundle；rollout observation；Iceberg maintenance | champion 不是現役模型（38）。沒有線上 shadow／canary 分桶 |

應保留的設計：離線內容處理；ingest 等 Kafka ack 後才接受；串流上傳；PIT fail-closed、不回退讀當前 Redis；artifact 原子 generation；shared feature assembler；release CAS；有界佇列與明確的 overload 回應。以下建議都以保留這些邊界為前提。

## 一、首版 23 項狀態總表

| # | 主題 | 狀態 | 主要剩餘缺口 |
| --- | --- | --- | --- |
| 01 | Flink 模式與啟動服務集合 | 部分解決 | Compose 已含 submitter；重新提交時不從 checkpoint 還原（25）；readiness 不看 job 與新鮮度；Helm 設為 flink 模式卻沒部署 Flink |
| 02 | content-worker 例外與 retry／DLQ | 部分解決 | `lease_held` 直接 ack，過期 run 沒人重新驅動；local 模式仍會刪除來源檔；CLIP／OCR 同步阻塞 event loop |
| 03 | 缺檔時建立模擬商品 | 部分解決（核心已修） | 舊 index 的產生者仍寫入合成向量；Helm 預設仍走舊 index；L2／IP 錯置（24） |
| 04 | HTTP 重送的 interaction identity | 部分解決 | payload_hash 含伺服器補上的 event_time，重送會變 409；前端每次產生新 UUID；ledger 沒有 retention |
| 05 | 曝光紀錄 served／viewed | 部分解決（主體完成） | viewed_at 用 client 時鐘（28）；outbox／views 沒有 retention；feature_context 沒有大小上限 |
| 06 | 上傳一致性 | 部分解決（核心完成） | 上傳沒有 idempotency key；COMMIT 結果不明時會刪除 object；reconciler 每次全量 list |
| 07 | Flink→Redis 寫入放大與一致性 | 部分解決 | 已改為單一 Lua 原子提交；仍逐筆同步 EVAL，沒有批次；key 沒有 hash tag |
| 08 | PIT 涵蓋整個模型閉環 | 部分解決 | Two-Tower 已走 PIT；Helm production 的 ranking 仍是 legacy，且永遠不會被評估；parity 沒有強制 |
| 09 | 離線指標與線上分數一致 | 部分解決 | 已共用 canonical score；NDCG 被稀釋（45）；fusion 無法離線評估；沒有 GAUC |
| 10 | 模型品質 gate | 部分解決 | 已有 lifecycle＋CAS；champion 不是現役模型（38）；小切片會阻擋（46）；沒有線上分桶 |
| 11 | 特徵表示 | 部分解決 | v3 categorical 只是 challenger；champion 仍是 hash/100；v3 沒有 item 表示 |
| 12 | Two-Tower holdout 與 sampling | 部分解決 | 已有 holdout、logQ、負樣本實驗；gate 分數方向錯誤（24）；seen／ranker_score 寫死（42） |
| 13 | 多路召回保留來源訊號 | 部分解決 | 已有 RRF＋retrieval_signals，但預設是 legacy；legacy 仍取 max 合併，且每類限 3 筆 |
| 14 | DIN／SASRec 分階段 | 未處理 | sasrec.py 沒有變更；production 沒有訓練路徑，且繞過 gate（51） |
| 15 | 多任務衝突與 MMoE | 未處理 | 仍是 shared-bottom＋固定權重，沒有 per-head 診斷 |
| 16 | 多模態消融與 encoding 重用 | 未處理 | 只新增了評估切片；trimodal 仍對每個候選重算 encoder |
| 17 | business value 與校準 | 部分解決 | 已有 label 契約與 temperature；預設 none；只有 temperature、沒有 bias；value 是幾何平均 |
| 18 | PIT reader／trainer 記憶體 | 部分解決（主體完成） | shadow 仍全量 read；retrieval 評估是 O(Q×N)（43）；verified generation 不會清除 |
| 19 | watermark、late data、狀態保留 | 部分解決 | 已有 idleness 與 side output；watermark 在 source 之後才指派；user state 沒有 TTL |
| 20 | 完整鏈路容量與 autoscaling | 未處理 | HPA 仍只看 CPU；coordinator 單副本；content task 只有 3 個 partition |
| 21 | CI、資料品質、演練 | 部分解決 | 前端、Flink、Helm、promtool 已加入 CI；integration 仍需手動觸發；品質 gauge 寫死（41）；Node 26 問題已解決 |
| 22 | 身分與資料存取範圍 | 未處理 | claims 只存不用；user_id 沒有限制；content status 沒有 owner 檢查 |
| 23 | 大檔拆分與 migration | 未處理（惡化） | system_store 5,433 行；migration 無法獨立建出 schema；Compose 沒有 production preset |

## 二、首版項目更新

### 01．部分解決：Flink 模式與啟動服務集合

**已完成。** `startup.sh:46` 以 `--profile flink` 啟動。submitter 會先檢查是否已有同名 job，有就不重送（`docker-compose.yml:1126-1152`）；`startup.sh:79-88` 檢查 job 恰好只有一個；`prometheus-rules.yml:283-325` 有 job missing／duplicate 與 checkpoint 告警。

**剩餘缺口。**

- 重新提交時，只有手動設定 `FLINK_RESTORE_SAVEPOINT_PATH` 才會加上 `-s`（`docker-compose.yml:1141-1150`）。JobManager 沒有設定 HA（`:1054-1066`），submitter 為 `restart: "no"`。後果見 25。
- gateway `/readyz` 不檢查 job、checkpoint 或特徵新鮮度。`/readyz/full` 反而在檢查 Flink 模式下根本不會執行的 feature-worker（`gateway/api.py:341`，見 32）。
- Helm 設定 `featurePipelineMode: flink`（`values.yaml:233`），但 featureWorker 關閉（`:852`），chart 也沒有部署 Flink。`kubernetes-deployment.md` 沒有把 Flink 列為外部依賴。Helm 的 topic 清單少了 `late-interaction-events`。

**驗收。** K8s 文件寫明 Flink job 的外部部署契約；readiness 與 alert 改依 job 是否存在、最近一次 checkpoint、`ufm:*` 新鮮度來判斷；`down/up` 後 generation 連續（同 25）。規模 S–M。

### 02．部分解決：content-worker 例外與 retry／DLQ

**已完成。** 失敗時會更新狀態並重新拋出，由 Kafka retry／DLQ 決定 offset。`(content_id, pipeline_version)` 有 claim、renew 與 owner-fenced complete（`system_store.py:2999-3252`）。worker 先寫入 durable artifact，再投影到 Redis 與向量；redelivery 時可以修復投影。

**剩餘缺口。**

- 當 claim 結果是 `lease_held`，程式直接 `return`（`video_processor.py:215-222`），consumer 隨即提交 offset，而且沒有任何流程會重新驅動「lease 已過期但仍是 processing」的 run。如果 worker 在處理途中被 OOM kill 或遇上滾動部署，而 redelivery 剛好落在 lease 期間內，訊息就會被 ack，job 永遠停在 processing。
- CLIP 與 PaddleOCR 在 event loop 內同步執行（`content_processor.py:1240,1306`）。在 `session_timeout_ms=10000`（`config.py:1917`）下，heartbeat 與 lease 續約都可能被餓死，進而引發 rebalance，再落入上一點的情境。
- local 模式下，`finally` 仍會刪除來源檔（`video_processor.py:463-470`），失敗、duplicate、superseded 路徑也一樣。第二次 retry 會得到 FileNotFoundError，DLQ replay 也沒有檔案可用。Compose 預設正是 local 模式。
- 只重試 3 次，退避為 250ms×n（`config.py:1900`），撐不過 S3 或 DB 的短暫中斷。
- 主模態推論失敗時，會靜默產生零向量並標記為 completed，見 54。

**建議與驗收。**

- `lease_held` 改為延遲重送（例如 retry topic），另外設一個 reaper，把過期的 processing run 重新排入佇列。
- 推論改到 `to_thread` 或 process pool 執行，確保 heartbeat 不中斷。
- 來源檔只在 completed 或 terminal 之後才刪除。
- 重試改用指數退避，並區分暫時錯誤與永久錯誤。

驗收：處理途中 kill worker，該 run 最終仍會 completed；第二次 retry 仍讀得到來源檔。規模 M。

### 03．部分解決（核心已修）：缺檔時建立模擬商品

**已完成。** 缺檔時改為 raise，不再產生 sample（`vector_search.py:144-160`）。production 強制 `VECTOR_BOOTSTRAP_MODE=required`（`config.py:326-334`），sample 只能用 `scripts/bootstrap_sample_vector_index.py` 明確產生。generation 以原子方式發布，並附 checksum manifest（`vector_search.py:732-827`）。

**剩餘缺口。**

- 舊 index 唯一的產生者 `load_dataset_from_csv`，仍用 `np.random`＋`hash(category)` 寫入合成向量（`data.py:1425-1449`）。換句話說，「真實 catalog」路徑產出的其實也是假向量。
- Compose 與 Helm 預設的 visual manifest 都是空的（`values.yaml:279`），所以仍走舊 index。Helm 的 `/app/models` 是 emptyDir，也沒有舊 index 的同步機制。
- 舊 index 是 L2，卻被當成相似度使用，見 24。content-worker 一律呼叫 `load_index()`，只部署 visual bundle 時會啟動失敗，見 53。

**建議。** 舊 index 退役，或讓它的產生者改用真實的 CLIP 商品 embedding；production 只接受 visual bundle。規模 S–M。

### 04．部分解決：HTTP 重送的 interaction identity

**已完成。** request 已有 `event_id`（`models.py:83`）。ingest 以 Postgres lease ledger 加 owner token 去重，同一個 key 搭配不同內容時回 409（`interaction_ingest/api.py:165-216`）。Flink closed-loop smoke 已驗證相同重送只會計算一次。

**剩餘缺口。**

- **bug**：`payload_hash` 包含 `request.event_time`（`interaction_ingest/api.py:165-176`），但 client 沒帶 event_time 時，validator 會用 `time.time()` 補值（`models.py:110-114`）。結果是：帶相同 event_id、但沒帶 event_time 的重送，會得到 409 而不是 duplicate。
- 前端 `api.js:76-90` 不送 event_time，而且每次呼叫都產生新的 UUID，所以前端的重試本身就不具冪等性。
- ledger 沒有綁定 caller scope；沒有 retention（見 31）。熱路徑上每筆互動要跑 2 個 Postgres transaction。

**驗收。** hash 只納入 client 實際提供的欄位（或規定帶 event_id 時必須帶 event_time）；前端在同一次使用者動作的多次重試之間保留同一個 event_id；測試 ack 遺失後重送，得到 duplicate。規模 S。

### 05．部分解決（主體完成）：曝光紀錄 served／viewed

**已完成。**

- served：先寫入 outbox，才回傳 impression_id（`recommendation/api.py:3037-3076`）。
- viewed：有 viewed API；前端以 IntersectionObserver 偵測 50% 可見並停留 1 秒（`App.jsx:719-730`）；ingest 會驗證 viewed 屬於某個 durable slate。
- Flink 有 views sink；PIT 只從 viewed 取負樣本（`PointInTimeFeatureJoinJob.java:344`）。

**剩餘缺口。**

- viewed／click 的 event_time 用的是 client 時鐘，PIT 的錨點因此偏移，見 28。
- `recommendation_event_outbox`、`interaction_idempotency_ledger`、`recommendation_impression_views` 都沒有 retention，見 31。
- `feature_context` 沒有裁切（`recommendation/api.py:2261`）。超過 aiokafka 預設的 1MB 時，`mark_failed` 會無限重試，沒有 terminal 狀態（`:2181-2205`）。
- outbox insert 在每個請求中同步 await，包括 cache hit 與 singleflight joiner，而且沒有 timeout，見 62。每個 uvicorn worker 各跑一個 dispatcher，每 0.25 秒查兩次 DB（`:3079-3091`）。
- 沒有 served→viewed→interaction 的覆蓋率指標，也無法依裝置或 cache path 分組檢視。

規模 M。

### 06．部分解決（核心完成）：上傳一致性

**已完成。** gateway 在同一個 DB transaction 內完成 stage 與 claim（`gateway/api.py:497-552`）。`content-task-publisher` 有 lease、穩定的 event ID、missing-object 終止與 orphan 清理。Kafka ack 之後才回傳 queued；否則回 503 `CONTENT_ENQUEUE_PENDING`。

**剩餘缺口。**

- 503 回應標記 `retryable:true`，但上傳沒有 idempotency key，client 重送會產生新的 content_id，同一支影片因此被重複處理。
- stage 的 COMMIT 結果不明時，`gateway/api.py:451` 會刪除 object，之後這個 job 被標記為 missing_object。
- `reconcile_objects`（`content_task_publisher/main.py:83-101`）每 5 分鐘全量列出 bucket 與所有 storage_path，記憶體為 O(N)。publisher 主迴圈沒有例外保護，DB 一抖動就會 crash。
- `video-processing-tasks` 只有 3 個 partition（`docker-compose.yml:1017`、`values.yaml:1061`），限制了 worker 的最大並行數。

規模 S–M。

### 07．部分解決：Flink→Redis 寫入放大與一致性

**已完成。** sink 改為單一 Lua 原子提交，一次完成 delta、trim 以及 epoch／generation／digest 更新（`InteractionFeatureJob.java:833-850`），不再逐項重寫整段序列。

**剩餘缺口。**

- 每筆事件都同步呼叫 `commands.eval`，每次都送出完整 script，沒有用 EVALSHA、pipeline 或批次。Redis 的 RTT 因此直接成為 sink 吞吐的上限。
- 5 個 key 都沒有 hash tag，搬到 Redis Cluster 會出現 CROSSSLOT 錯誤。
- generation 只在 job state 內遞增，空 state 重新提交時會倒退，見 25。parallelism>1 時，先 `keyBy(eventId)` 再 `keyBy(userId)`，同一 user 的事件順序不固定，也可能觸發 conflict。

**驗收。** 量測每個事件的 Redis RTT 數、sink p95、checkpoint 時間；在 Cluster 模式下跑整合測試。規模 M。

### 08．部分解決：PIT 應涵蓋整個模型閉環

**已完成。** Two-Tower 改由 retrieval PIT 訓練（`scripts/train_retrieval_pit.py`）。production 強制 `RETRIEVAL_TRAINING_SOURCE=pit`，並關閉 online retraining（`config.py:2876-2887`）。

**剩餘缺口。**

- production 沒有強制 ranking 走 PIT。Helm 同時設定 `environment: production` 與 `trainingSource: legacy`（`values.yaml:144,216`）。legacy adapter 讀的是當前 Redis（`legacy_training_adapter.py:43`），而 legacy release 永遠不會被評估（`model_trainer/main.py:1113` 的條件需要 `use_pit_dataset`），所以在 enforced gate 下進不了 staging。設定矩陣因此自相矛盾。
- online/offline parity 只記錄、不強制，見 41。
- v1 契約從本機權重 warm-start，而且沒有 lineage（`ranking.py:2546-2584` 只有 v3 會 reset）。manifest 沒有記錄 window、lateness 與來源表 snapshot，無法只憑 manifest 重現資料集。
- retrieval PIT 沒有排程，catalog generation 要手動填寫（`deploy/compose/pit-production.conf:9`）。DIN 經 PIT 拿不到資料（34）；SASRec／Swing 等召回來源沒有 PIT 訓練路徑（51）。

**建議。** 明確列出 demo、legacy rollback、PIT 三種 preset，production 只接受 PIT；manifest 補齊重現所需的參數。延續 168 小時 attribution 與 lateness，不要為了快速出模型而縮短窗口。驗收：未來特徵零洩漏、相同 manifest 可重現、online/offline feature parity 達標。規模 L。

### 09．部分解決：離線排序指標與線上分數一致

**已完成。** LTR 與 gate 的 NDCG 改用 canonical serving score 計算（`ranking.py:4091`、`model_release_quality.py:653-680`）。release 預測會先套用 calibration 再計分（`ranking.py:3161-3200`）。另外新增 PR-AUC、log loss、Brier、ECE、value WAPE，並在 holdout 上做 ONNX parity（`scripts/evaluate_phase_c.py:208-244`）。

**剩餘缺口。**

- NDCG 被全 view 的 slate 稀釋，見 45。
- 評估只重排 logged slate，無法評估候選集的變更。`train_phase_c.py:99-110` 的 fusion challenger 直接複製 champion 權重，NDCG Δ 恆為 0，只能依賴外部提供的 Recall@500。
- 沒有 GAUC 或 per-user 指標。
- business score 開啟時，`ranking_tower` 沒有任何 loss。首版建議保留的 ranking-head 診斷，實際上是未經訓練的輸出。
- LTR 訓練用的是未校準的 ctcvr，serving 用的是校準後的 ctcvr×value，兩者的排序可能不同。

規模 S–M。

### 10．部分解決：artifact 可載入不等於模型值得上線

**已完成。**

- 有 registered → validated → staging → active 的 lifecycle 與 CAS（`system_store.py:4762-5195`），activation 需要 passed evaluation（`:5029-5046`）。
- 有 user-cluster paired bootstrap（`model_release_quality.py:683`），以及 cold、長尾、價格、類別、來源、模態等切片（`:501-560`）。
- gate 設定已凍結（`phase_c_evidence.py:11-27`）。

**剩餘缺口。**

- trainer 的 champion 不是現役 release，見 38。activation 也不會檢查 `evaluation.champion_release_id` 是否仍是現役。
- 任一小切片證據不足，整體就判 INSUFFICIENT，見 46。只有 NDCG 有 CI，其他指標都是點估計門檻。bootstrap 無法擴展到真實資料量，見 50。
- 沒有線上 shadow／canary、實驗分桶、assignment version、SRM 檢查或 propensity 紀錄。
- Phase C 的 recall evidence 是外部 JSON，只驗證 lineage 欄位（`evaluate_phase_c.py:64`）。trainer 的 gate 路徑沒有測試。

規模 M–L。

### 11．部分解決：特徵表示

**已完成。** v3 vocab 含 PAD／MISSING／OOV、multi-hot mean pooling 與 4 個 cross（`ranking_categorical.py:235-506`）。normalization 採 p1/p99 clipped z-score 加 missing indicator，只用 training split 擬合（`ranking_categorical.py:77-160`）。

**剩餘缺口。**

- v3 只是 challenger，Compose 預設仍是 `ranking_ltr_v1`（`docker-compose.yml:136`）。champion 的 brand 仍是 hash/100，tags 與偏好仍只取長度（`ranking.py:680,713`）。
- 未縮放的天數特徵直接進入 row-wise LayerNorm（`dcn.py:156`），一個大值就會把其他特徵壓扁。
- v3 禁止搭配 DIN／trimodal（`config.py:1631-1636`），也沒有 product-ID embedding 或 content projection。candidate 的原生分數被清零（`ranking_features.py:107-109`），導致同類、同品牌、同價位的商品幾乎無法區分，冷啟動也只能依賴 metadata。
- session intent×item 尚未實作，也沒有消融證據。

**建議。** 加入有頻率門檻的 product-ID embedding（含 OOV），或把 Two-Tower／CLIP sidecar 接一層小型 projection，再逐項消融。驗收看 serving NDCG@K、冷使用者切片、校準與 p95。規模 M。

### 12．部分解決：Two-Tower holdout 與 sampling

**已完成。**

- 7 天時間 holdout（`retrieval_pit_dataset.py:735-751`）。
- full-catalog Recall@50/100/200、MRR、切片與 user bootstrap（`retrieval_evaluation.py`）。
- logQ 修正（`two_tower.py:224-248`）。
- ranker_rejected 分成 disabled／weak／teacher_soft 三組實驗（`retrieval_training.py:224-260`）。

**剩餘缺口。**

- gate 把 L2 距離當相似度，結論失真，見 24。
- logQ 預設關閉（`config.py:773`）。
- seen 與 ranker_score 在 Flink 端寫死，teacher_soft 的目標恆為 0.5，見 42。
- 評估是 O(Q×N)，見 43。
- 快照之後才出現的新用戶全部共用 ID 0；外部負樣本永遠取最舊的幾筆，見 55。

方法依據：[Sampling-Bias-Corrected Neural Modeling](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) 與 [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/)。它們支持實驗設計，但不構成本專案的效果證據。規模 M。

### 13．部分解決：多路召回應保留 source-specific 訊號

**已完成。** `candidate_fusion.py` 實作 RRF（k=60，最多 500 筆），保留各路的 `retrieval_signals`（raw score、rank、normalized rank）。RRF 模式沒有前置類別限額，lineage 會寫入 impression。

**剩餘缺口。**

- production 預設仍是 legacy（`config.py:357`、`docker-compose.yml:141`、`values.yaml:342`）：取 max 合併（`recommender.py:2250-2264`）、固定權重（`:2924`）。Swing 是未正規化的加總，量尺壓過 Two-Tower 的 [0,1] 分數。
- `_apply_diversity_filter` 預設每類只留 3 筆（`config.py:636`），類別用完就 break（`recommender.py:3048`）。照原邏輯模擬：6 個類別、320 個候選，最後只剩 21 個進入精排。
- RRF 沒有來源權重或 quota，`random` 與 Two-Tower 同權。RRF 只看名次，因此冷啟 confidence 的打折（`recommender.py:1006`）沒有作用。category pool 依偏好類別串接，排在後面的類別名次被系統性壓低（`:2664-2690`）。

**建議。** 先把 legacy 的前置類別限額改成只在 slate 階段生效。RRF 加上來源權重／quota，用 shadow 方式比較。監測每一路的 unique recall、進入 Top-K 的比例，以及移除某一路後的增量損失。規模 M。

### 14．未處理：DIN／SASRec 依目標分階段啟用

**現況。**

- `sasrec.py` 自首版後沒有變更：
  - 全 catalog logits（`:128`）。
  - 每位使用者只取最後一段 window（`:262`）。
  - full softmax CE（`:198`）。
  - serving 計算 L×N logits，但實際只需要最後一個位置。
- 左對齊 padding 讓 position 的語義隨序列長度改變；沒有時間 holdout。
- production 沒有 SASRec 訓練路徑，而且透過 `sync_latest` 載入、繞過 release gate（`recommender.py:1619`），見 51。
- DIN 的分階段消融沒有執行紀錄。DIN 本身另有 batch 統計問題（49）與經 PIT 拿不到資料的問題（34）。

**首版建議仍適用。** 加入時間間隔、action／session boundary、多個 sequence window、sampled loss，以及 sequence representation→ANN。全 catalog logits 的理論記憶體為 `B × L × N × dtype_bytes`：例如 B=256、L=50、N=100 萬、fp32，光 logits 就約 51.2 GB。驗收看 next-item Recall@K、session 切片、item coverage、latency 與 memory。規模 M，ANN 改造 L。

### 15．未處理：多任務架構先量測衝突，再評估 MMoE

**現況。** 架構仍是 shared-bottom＋task towers（`ranking.py:152-296`），loss 權重固定（`:4083-4089`）。training metrics 只記錄總 loss（`:3135-3160`），沒有 per-head loss、梯度 norm 或 cosine。ListNet 改用 canonical score 後，反而多了一條跨任務耦合，見 47。`cvr_weight=2.0`（`config.py:1302`）已被 `direct_cvr_weight` 取代，是無作用的設定。

**建議不變。** 先記錄各 head 的 loss、梯度 norm／cosine、正樣本數與校準，確認是否有負遷移；有的話，再以相同資料、參數量與 serving budget 比較 MMoE／PLE。方法依據：[MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) 與 [DCN V2](https://arxiv.org/abs/2008.13535)。是否適合本資料仍須實驗。規模 M–L。

### 16．未處理（只有評估切片有進展）：多模態消融與 video encoding 重用

**現況。**

- `content_processor.py` 與 `temporal_multimodal.py` 自首版後沒有變更；CLIP B/16＋e5-small 只是 pin 了 revision（`config.py:88-93,245-252`）。
- serving 用 `np.repeat` 把整支影片的 token 複製給每個候選（`ranking.py:1954-1966`），三個 temporal encoder 對 B 份相同序列各算一次（`:557-578`）。
- trimodal 不支援 ONNX（`ranking_onnx.py:105`）。Phase C 強制關閉 trimodal（`train_phase_c.py:87`），因此 gate 的 `missing_*` 模態切片在 Phase C 永遠是 not_applicable（`model_release_quality.py:260`）。
- `add_content_embedding` 仍是 placeholder（`vector_search.py:550-562`），worker 卻仍在呼叫它。
- retrieval 評估已有 missing-modality 切片，但沒有 ablation runner。

**建議不變。**

- 依序消融：mean CLIP → learned pooling → OCR → ASR → candidate-conditioned fusion。
- 與候選無關的 token encoding，以 content hash＋encoder version 為 key 快取重用。
- 大模型 teacher、商品實體抽取、domain LoRA 都放在離線，通過檢驗後才蒸餾或升級索引。
- OCR 語系問題見 56。

規模 M–L。

### 17．部分解決：business value 與校準

**已完成。**

- `gross_revenue_v1` label 契約：source、currency、conflict 都會 mask（`ranking_training.py:139-160`）。
- 校準使用獨立的 pre-holdout split（`pit_streaming.py:288-303`）。
- temperature 擬合（`ranking.py:2746-2795`）與 serving 套用（`:2016-2021`）。
- gate 有 ECE 與 reliability table。

**剩餘缺口。**

- 生產預設 `probability_calibration=none`（`docker-compose.yml:139`）。
- 只有 temperature、沒有 bias，修不了 base-rate 偏移；搭配 `ctcvr_pos_weight`（`config.py:1311`）時必然失準。應改試 Platt(a,b) 或 isotonic。
- CTR 與 CTCVR 分開校準，CVR 則完全沒有校準，可能出現校準後 ctcvr > ctr 的情況（`ranking.py:2010-2023`）。
- value head 在 log1p 空間用 Huber 回歸，再以 expm1 還原（`ranking.py:3951-3980`），得到的近似幾何平均（接近中位數），而不是 E[v|purchase]，價值分散大的商品會被系統性低估。可以用 smearing（×exp(σ²/2)），或改用 Tweedie／Gamma loss，並在 gate 加上分價帶的 Σpred/Σactual。
- ListNet 會污染 value head，見 47。
- 沒有「每千次實際曝光的 value」、延遲轉換或退款 guardrail。

規模 M。

### 18．部分解決（主體完成）：PIT reader 與 trainer 的全量記憶體路徑

**已完成。** 兩階段 streaming reader：pin 住 manifest、分批驗證 shard、以 SQLite spool 保留完整的 impression／query group（`pit_training_dataset.py:355-493`、`pit_streaming.py:434`）。value stats、vocab 與 dense normalization 都只 fit training split。PROGRESS 記錄 100k／500k 列 benchmark 的 peak RSS 約 274MB。

**剩餘缺口。**

- trimodal shadow 預設開啟，每次訓練都會再跑一次全量 `read()`（`model_trainer/main.py:1215-1221,1362`），見 36。
- retrieval 評估是 O(Q×N)，見 43。`verified/ranking-<sha>` generation 永遠不會清除（`pit_streaming.py:203`），見 39。
- `pit_streaming.py:434` 每個 epoch 都用 UDF 對含 payload 的整張表排序；應改為先排 group key，再用索引取列。
- `pit_memory_benchmark.py` 只測 spool 加一個純量模型，沒涵蓋真實的 RankingModel、validate_and_stage 或 retrieval 評估。

規模 M。

### 19．部分解決：Flink watermark、late data 與狀態保留

**已完成。** 已設定 `withIdleness`（`InteractionFeatureJob.java:151`）、allowedLateness＋`sideOutputLateData`＋quarantine topic（`:248-262`），並驗證 dedup TTL ≥ replay 範圍＋1 天（`:1372-1381`）。

**剩餘缺口。**

- watermark 在 `fromSource(..., noWatermarks())`（`:93`）之後才指派（`:148-152`），所以沒有 per-partition watermark。在 6 個 partition、parallelism 1 的設定下追趕 backlog 時，稀疏 partition 會把 watermark 往前推，密集 partition 的事件就被判為 late，視窗特徵因此被系統性低估。應把 WatermarkStrategy（含 idleness）放進 `fromSource`，必要時加上 watermark alignment。
- user state 沒有 TTL 或 compaction 策略。`active_keys` 是 Counter，restore 後會從 0 重算（`:692`）；bytes gauge 只反映最後處理的那位 user（`:695`），不能拿來估算容量。
- materializer 的 dedup TTL 為 400 天（`FeatureHistoryMaterializerJob.java:795`），state 沒有上界。

**驗收。** 加一個「停 job 1 小時再追趕」的 drill，斷言 late rate；state bytes/user 改為全量統計。不要直接加 TTL，否則 lifetime count 會無聲重置。idleness 的原理與 API 參考 [Flink 1.20 官方文件](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/event-time/generating_watermarks/)。規模 M。

### 20．未處理：先取得完整鏈路容量，再決定服務拆分與 autoscaling

**現況。**

- HPA 仍只看 CPU（`charts/video-commerce/templates/hpa.yaml:47-53`），沒有 KEDA 或 external metrics。
- `rankingCoordinator` 為單副本，HPA 與 PDB 都關閉；contentWorker 也是單副本，HPA 關閉；`video-processing-tasks` 只有 3 個 partition。
- 每個 recommendation process 仍各自複製一份 catalog 與 ANN，只有 RSS gauge（`observability.py:1570`）。
- 新出現的 DB 連線預算問題見 62；coordinator 透過 readiness 變成全站單點，見 59。

首版的容量量測建議不變，Triton A/B 也仍未在固定的 Linux 主機上完成。另外要注意：多 worker 指標失真（61）修好之前，所有基於 Prometheus 的容量結論都要重新核對。規模 M–L。

### 21．部分解決：CI、資料品質監控與操作演練

**已完成。**

- CI 加入前端 vitest（`ci.yml:39-40`）、Flink Maven（`:46-60`）、Helm lint/template＋kubeconform＋promtool（`:62-75`），以及 Flink closed-loop 與 recovery drill（`:115-212`）。
- Node 26 的 Web Storage 問題已由 `Code.Frontend/src/test/setup.js:4-13` 的 shim 解決；本次在 Node 26.7.0 跑 vitest，14 項通過。
- PIT parity、leakage 與 label reconciliation 已有 alert（`prometheus-rules.yml:426-471`）。

**剩餘缺口。**

- HTTP integration smoke 仍只在 `workflow_dispatch` 時執行（`ci.yml:107-113`），Playwright e2e 不在 CI。Helm 實際使用的 `content-worker`、`pit-orchestrator` image target 沒有在 CI build（`:77-86`），這正是 68 沒被抓到的原因。
- promtool 只測 60 條 alert 中的 6 條。PIT parity／leakage gauge 是寫死的值，相關 alert 永遠不會觸發，見 41。
- 沒有線上 feature／prediction drift、依模型版本的 calibration，或各召回來源貢獻的監控。
- 演練只涵蓋有計畫的 savepoint，沒有 JobManager 非預期 crash（25）、多 instance rolling activation，也沒有對真實 REST／S3 執行 Iceberg 清理。
- `package.json` 寫 `engines >=16`，但 jsdom 29 要求 Node `^20.19`。vitest 4 內含 vite 8，production build 卻用 vite 5.4，測試與 build 的 bundler 大版本不同。
- Python 沒有 lockfile（見 71）。CI 沒有 `concurrency`、沒有 pip cache，多數 job 也沒設 timeout（見 75）。

規模 S–M。

### 22．未處理：身分與資料存取範圍

**現況。**

- gateway 驗證 bearer 之後只把 claims 存起來（`gateway/api.py:102`），轉送的仍是 raw body（`:772`）。上傳的 `user_id` 走 query 參數（`:389`）。content status 與 analytics 沒有 owner 檢查（`:561,578`）。
- `user_id` 沒有任何字元或長度限制（`models.py:56,86`），衍生出 glob 注入（29）與 JDBC 卡死（26）。
- 前端不帶任何憑證（`Code.Frontend/src/api.js:4-10`），但 production 強制要 API key。結果是前端在 production 會收到 401；若要能用，就得把 key 放進前端。
- bearer 模式另有 JWKS 同步抓取，以及 aud／iss 未強制的問題，見 65。

**建議。** 首版建議不變，另外補充：所有 public ID 欄位統一加長度與 regex 限制；前端改走 session、BFF 或短期 token。規模 M，完整多租戶 L。

### 23．未處理（惡化）：以契約為邊界拆分大檔案

**大檔。** 從首版到現在：`system_store.py` 3,075→5,433 行、`ranking.py` 3,969→4,633 行、`recommendation/api.py` 3,150→3,443 行、`config.py` 2,778→3,165 行、`App.jsx` 1,820→1,868 行。新功能多半另開新檔，但原有的 monolith 仍持續變大。

**Migration。**

- 001–011 的 SQL 已經存在，但沒有 runner，也沒有 version table；002–011 沒有包在 transaction 裡。
- Compose 與 Helm 仍設 `AUTO_CREATE_SCHEMA=true`（`docker-compose.yml:66`、`values.yaml:124`），每個 process 啟動時都會跑 `create_all` 加上 index DDL（`system_store.py:1374-1391`）。
- migration 本身無法獨立建出 schema：`content_jobs`、`model_checkpoints` 只由 `create_all` 建立，但 `006:30`、`009:1`、`010:3` 會 ALTER 或 REFERENCE 這兩張表。
- 每月的 partition 只在啟動時建立（`system_store.py:1552-1571`）。一旦關掉 startup DDL，資料會落進 DEFAULT partition。

**demo／prod 分離。**

- production 已禁用 sample bootstrap，fail-fast 檢查也已擴充（`config.py:331-335,2849-2894`）。
- 但 Compose 預設 `ENVIRONMENT=development`；`startup.sh:33` 說這是 production stack，實際卻跑 demo profile（`:45`）。
- 若把 Compose 設成 production，又會因 `MODEL_RELEASE_GATE_MODE` 預設為 observe 而 fail-fast，`pit-production.conf` 也沒有覆寫這一項。

**設定規模。** `config.py` 有 597 個 `Field`；`.env.example` 有 427 個 key，其中 10 個重複（例如 `FEATURE_LAKE_ENABLED` 出現兩次，編輯前面那一個不會生效）；另有 45 處直接呼叫 `os.getenv`，繞過了 `*_FILE` 機制。

**建議。**

- 先補 migration runner、version table 與 partition 維護 job，之後才關閉 startup DDL。
- `system_store.py` 依 outbox、lease、release registry、PIT run 拆成獨立 repository。
- 提供一個可以通過 production 驗證的 Compose preset（見 70）。
- 自動產生 env 文件，並加入 config／Compose／Helm 的 parity 測試。
- 每次拆分都保持 API、feature vector、artifact checksum、cache key 與排序結果可比較。

規模 M，分批完成。

## 三、新發現

### 3.1 P0

### 24．P0：FAISS metric 錯置，Two-Tower release gate 把 L2 距離當成相似度

**證據。**

- `two_tower.py:1676` 以 `faiss.IndexHNSWFlat(self.output_dim, 32)` 建立 index，FAISS 的預設 metric 是 `METRIC_L2`；`write_two_tower_bundle` 把這個 index 寫入 release bundle（`retrieval_release_training.py:273-274`）。
- `score_ann_index` 直接存原始的平方距離（`retrieval_evaluation.py:136-140`），`evaluate_full_catalog` 卻依 `-score` 由大到小排序（`:180-186`）。champion 走同一條路徑（`scripts/train_retrieval_pit.py:62`）。
- serving 端的 `_cf_search_score_to_similarity`（`recommender.py:1117-1127`）會依 metric 轉換，註解也寫明是 L2，所以只有評估路徑出錯。
- 現有測試使用回傳內積分數的 fake index（`tests/test_retrieval_evaluation.py:146-166`），因此沒抓到。

**同類問題。** 舊 content index 也用 L2 的 `IndexHNSWFlat`（`vector_search.py:304`）。搜尋後以 `score < similarity_threshold`（0.1）過濾，並把距離直接當成 `content_similarity_score`（`:475-485`）。結果是：最近的商品（距離≈0）反而被門檻刪掉，距離越遠分數越高。visual manifest 為空時（也就是 Compose／Helm 的預設）走的正是這條路徑。

**影響。**

- ANN 取回的 top-200 集合是對的，但順序顛倒。Recall@50/100 與 MRR 量到的，其實是 top-200 裡最遠的那一批。
- 和排序正確的 popularity baseline 比較時，challenger 會系統性 FAIL。有 champion 時兩邊同時反轉，PASS／FAIL 都沒有意義。`exact_audit_ann_recall` 也一起失真。
- 舊 content index 讓 legacy 融合（`content_weight=0.3`）把不相似的商品往前推，ranker 收到的 content 特徵語義也是相反的。

**建議與驗收。**

- bundle 改用 `METRIC_INNER_PRODUCT`（與 `visual_product_index.py:178` 一致；向量本來就已 L2 normalize，見 `two_tower.py:427,504`），或讓 `score_ann_index` 依 `metric_type` 轉換。
- 舊 content index 比照處理，並拒絕載入沒有標示 metric 的舊 generation。
- 補一個使用真 FAISS HNSW 的端到端測試，要求 ANN audit@50 ≥ 0.95，且 challenger 與 champion 相同時 Δ≈0。
- 修正之前產生的 retrieval evaluation 一律視為無效。

規模 S；信心 高（已讀原始碼確認）。

### 25．P0：Flink job 不帶 savepoint 重新提交後，Redis 寫入全被判 stale，接著 conflict 造成 crash loop

**證據。**

- Lua 提交邏輯：`epoch` 相同但 `generation` 較小時回傳 `stale`；兩者相等但 digest 不同時回傳 `conflict`（`InteractionFeatureJob.java:840-841`），而 `conflict` 會直接丟出 `IllegalStateException`（`:897-898`）。
- generation 只存在 job state 中（`:1521`）。epoch 預設為 1（`:1342`），只有 production 才要求明確設定（`:1366-1372`）。文件規定只有「刻意放棄 state」時才 bump epoch（`flink-realtime-recovery.md:3-7`）。
- 但例行操作都會在不 bump epoch 的情況下放棄 state：
  - `startup.sh restart` 就是 down＋up（`startup.sh:55-58`）。
  - submitter 只有在手動設定 `FLINK_RESTORE_SAVEPOINT_PATH` 時才加 `-s`（`docker-compose.yml:1141-1150`）。
  - JobManager 沒有 HA（`:1054-1066`）。
  - rollback runbook 用的是 `--profile flink up -d --build`（`deploy-rollback-runbook.md:40`）。
- Redis 以 volume＋AOF 持久化，`ufm:*` 的 TTL 是 30 天，所以舊的 generation 紀錄一直都在。

**影響。**

1. 重新提交後，所有既有使用者的 generation 都從 1 重新開始，寫入一律被判為 stale。`uf:` 在 30 分鐘後過期、退回 default，序列也停止更新。serving 拿到的是退化的特徵，卻不會報任何錯誤。
2. 只要有任一使用者的 generation 追平舊值，因為 digest 不同就觸發 conflict，task 失敗；從 checkpoint 重播後又遇到同一筆，形成無限 crash loop。官方特徵、Postgres 寫入與 impression sink 全部停擺，lifetime counters 也同時歸零。
3. recovery drill 只測 savepoint 路徑（`run_flink_recovery_drill.sh:62-64`），抓不到這個情境。

**建議與驗收。**

- submitter 預設從最新的 retained checkpoint／savepoint 還原。找不到可還原的狀態、而 Redis 已有同一 epoch 的 `ufm:*` 時，拒絕啟動，或強制 epoch＋1。
- JobManager HA：Flink 1.20 standalone 沒有檔案型 HA，只能選 ZooKeeper／K8s HA 或 application mode。Compose 改由 submitter 自動從 retained checkpoint 還原（Phase 0 採用此作法）。
- conflict 改為隔離並記錄 metric，不讓 job crash；並加上 stale rate 告警。

驗收：在 `startup.sh restart`、Flink image 重建、kill JobManager 三種情境下，generation 都連續、conflict 為 0、lifetime counters 不歸零。規模 M；信心 高（已讀原始碼確認）。

### 3.2 資料管線（線上）

### 26．P1：單一超長欄位或重複使用的 event_id，會讓 Flink JDBC sink 無限重試

**證據。**

- 欄位長度：gateway／ingest 直接採用 client 傳來的 `X-Request-ID`（`service_common.py:280`）。`user_id`、`product_id`、`content_id` 都沒有 `max_length`（`models.py:56-57,86-87`），但 DB 欄位是 VARCHAR(64)/(255)（`system_store.py:637-667`）。
- 重試：Flink JdbcSink 重試到 `withMaxRetries` 上限後就丟出例外（`InteractionFeatureJob.java:274-341`）。
- 衝突目標不一致：views sink 用 `ON CONFLICT (impression_id, product_id) DO NOTHING`（`:391`），但該表的 PK 是 event_id（migration 008），ingest 也不檢查同一請求內各 item 的 event_id 是否唯一。
- ingest 端的 `on_conflict_do_nothing()` 沒有指定 target，同樣的衝突在那裡被靜默吞掉。

**影響。** 只要一個請求帶 65 字元的 request id、256 字元的 product_id，或重複使用某個 view event_id，整批寫入就會失敗。job 重啟後重播同一筆、再次失敗，官方特徵因此全部停止更新，而 ingest 端看不出任何異常。

**建議與驗收。**

- gateway／ingest 對所有 public ID 限制長度與格式；request id 不合法時，由 server 重新產生。
- Flink 在 parse 階段遇到超長欄位就直接送 DLQ。
- views sink 的 conflict 處理改為與 PK 一致。

驗收：各種超長欄位與重複使用的 event_id 都只會進 DLQ，job 不會重啟。規模 S–M；信心 高。

### 27．P1：官方 `uf:` 的 TTL 只有 30 分鐘，PIT 卻不設特徵年齡上限，造成 train/serve skew

**證據。** 官方 `uf:` 的預設 TTL 為 1800 秒（`InteractionFeatureJob.java:1350`，由 Lua 以 `SETEX` 寫入），但序列、`ufm`、`ust` 都是 30 天。讀不到 `uf:` 時，`feature_store.py:470-487` 回傳 default。PIT 的 user_ranked 只取 as_of 之前的最新一筆，不論有多舊（`PointInTimeFeatureJoinJob.java:459-465`）。

**影響。** 使用者閒置 30 分鐘後第一次請求推薦時，總互動數、CTR、CVR、偏好類別全部是 0 或空，序列卻還在；同樣的情境在訓練資料裡卻是真實值。這是 online/offline 不一致，而且剛好落在「回訪使用者」這個重要切片上。

**建議與驗收。** 官方 key 的 TTL 與 `ufm` 對齊，或與 PIT 共用同一份 max-age 契約；serving 端加上「已知使用者特徵缺失」的 metric。驗收：閒置超過 30 分鐘的使用者，online/offline 的 bundle hash 一致。規模 S；信心 高。

### 28．P1：client 時鐘被當成 PIT 錨點

**證據。**

- client 時間的來源：前端以 `Date.now()/1000` 作為 viewed_at（`Code.Frontend/src/api.js:128`），click 也帶 client 時間。`event_time.py:7` 允許的範圍是過去 30 天到未來 5 分鐘；`kafka_client.py:987` 把 event_time 直接設為 viewed_at。
- server 時間的來源：observation 用的是 server 時間（`interaction_ingest/api.py:157,334`）。
- ranking PIT 要求 view／click 的 event_time ≥ server 的 as_of（`PointInTimeFeatureJoinJob.java:477-487`）。
- retrieval PIT 以 view 的 event_time 作為 as_of（`RetrievalPointInTimeJoinJob.java:79`），要求特徵的 available_at ≤ as_of（`:102`），並要求 feedback 的 event_time ≥ as_of（`:117`）。

**影響。**

- client 比 server 慢幾秒：該裝置的所有曝光與點擊都進不了 ranking PIT，形成依裝置而異的選擇偏差。
- client 比 server 快：retrieval 那一列的 user 特徵已經包含這次點擊，label 卻因為點擊時間早於 as_of 而被標成 `viewed_no_positive`，同時造成洩漏與錯標。

**建議與驗收。** ingest 把 event_time clamp 到 `[served_at, received_at]` 的範圍內，另外保存 client 原始時間；PIT join 改用 server 時間；並量測時鐘偏差的分佈。驗收：注入 ±60 秒偏移的 fixture，PIT 的行數與 label 都不變，leakage 為 0。規模 S–M；信心 高。

### 29．P1：每筆互動都對 Redis 做全 keyspace SCAN；known-user 名單上限 20 萬；user_id 可做 glob 注入

**證據。**

- ingest 對每筆非 duplicate 的互動，以 fire-and-forget 方式呼叫 `invalidate_user_serving_cache`（`interaction_ingest/api.py:295-299`）。這個函式用 `scan_iter(match="rc:{user_id}:*")` 與 `cc:{user_id}:*` 掃描（`feature_store.py:1799-1810,2122-2146`），timeout 為 0.25 秒。
- gateway 每次被 `/metrics` scrape 時，都對主 Redis 跑 3 次 heartbeat SCAN（`gateway/api.py:673-681`、`feature_store.py:1956`）。
- known-user snapshot 每個 worker 每 60 秒 SCAN 一次 `uf:*`／`ust:*`，到 20 萬筆就 break（`feature_store.py:397-402`）；不在名單內的使用者直接跳過 Redis（`:296-301`）。
- `user_id` 沒有任何字元限制。

**影響。**

- **效能與失效**：SCAN MATCH 的成本與總 key 數成正比。百萬 key 時每筆互動要跑上千次 SCAN，0.25 秒逾時後 cache 根本沒清掉，使用者繼續拿到舊推薦，Redis CPU 卻持續被吃掉。
- **注入**：`user_id="*"` 可以清空所有使用者的推薦與候選 cache，引發 stampede；含 `:` 的 user_id 會誤刪別人的 cache。
- **名單上限**：活躍使用者超過 20 萬後，名單外的人拿到的是 default 特徵與空序列，而且沒有任何 metric 會顯示。
- 另外，cache key 已經包含 sequence token，這個 invalidation 多半是多餘的。

**建議與驗收。**

- 移除 SCAN invalidation，改為每位使用者一個 cache generation（`INCR` 後併入 cache key）。
- heartbeat 改存成單一 HASH 或 ZSET。
- known-user 改用 Bloom filter，或移除上限並加上「達到上限」的 metric。
- 限制 user_id 的 regex。

驗收：百萬 key 的情況下，每筆互動的 Redis 命令數為常數。規模 M；信心 高。

### 30．P1：window feature 的合約斷裂

**證據。**

- 產生端：Flink 對 user、product、category 三種 entity 產生 window 特徵（`InteractionFeatureJob.java:705-713`），並原樣送進 feature-history（`:1057-1075`）。
- 物化端：materializer 只接受 `window_feature`＋`user`，其他組合一律丟例外、送進 DLQ（`FeatureHistoryMaterializerJob.java:446-457`）；PIT 也沒有 join window feature history。
- 讀取端：Redis 的 `rtwf:` TTL 為 86400 秒，沒有新事件就不會產生新的 window；讀端（`ranking.py:776-817`）不檢查 window_end。

**影響。**

- feature-lake profile 下，每個活躍商品或類別每分鐘都產生一筆 DLQ。`FeatureLakeDlqTraffic`（`prometheus-rules.yml:390`）因此一直響，蓋掉真正的合約錯誤。
- 如果啟用 realtime window features，訓練資料全是 0，線上用的卻可能是 24 小時前的「5 分鐘」計數。

**建議與驗收。** materializer 接受 product／category；PIT 加上 window join；讀端在 `now - window_end > slide` 時視為 0。驗收：DLQ 為 0，並有 window parity 測試。規模 M；信心 高。

### 31．P2：事件表無上界成長

**證據。** `system_store.py:1653-1705` 的清除邏輯只處理 interaction_events 與 impressions。以下三張表都沒有清除：

- `interaction_idempotency_ledger`：每筆互動一列，含 JSONB。
- `recommendation_event_outbox`：每次推薦一列，含 slate 與特徵 snapshot。
- `recommendation_impression_views`。

**影響。** 表的大小隨 QPS 線性成長，vacuum 與索引成本也跟著上升。日後補上 retention 時，如果 ledger 保留不到 30 天（也就是 public 事件的接受窗口），重送的事件會被重複計數。

**建議與驗收。** 以分區或分批方式清除：ledger 至少保留 30 天再加緩衝，outbox 只保留 published 後 N 天；並監控表的大小。規模 S；信心 高。

### 32．P2：Flink 模式下 feature-worker 陷入重啟迴圈，造成告警誤報

**證據。**

- `feature_updater.py:512-516` 在 flink 模式下直接 return，而 Compose 設定 `restart: unless-stopped`（`docker-compose.yml:340`），所以它會不斷重啟。
- Prometheus 仍持續 scrape 它（`monitoring/prometheus.yml:49`），`WorkerMetricsTargetDown`、`WorkerHeartbeatMissing`（`prometheus-rules.yml:81-97`）因此永久觸發。
- gateway 的 `/readyz/full` 仍檢查它（`gateway/api.py:341`），而 K8s 文件要求執行 `curl -f .../readyz/full`（`kubernetes-deployment.md:208`）。
- `slo-and-alerting.md:6` 以 feature-worker heartbeat 作為新鮮度 SLO，量的是錯誤的元件。

**影響。** 預設部署下有兩個 critical alert 一直在響，`/readyz/full` 永遠回 503，真正的 Flink 故障反而被淹沒。

**建議。** 依 `FEATURE_PIPELINE_MODE` 決定是否啟動 feature-worker，以及是否啟用對應的 alert；新鮮度 SLO 改看 Flink checkpoint 與 `ufm` 的更新時間。規模 S；信心 高。

### 33．P2：有狀態 operator 沒有固定 uid；EXACTLY_ONCE 名不副實

**證據。**

- 兩個 KafkaSource（`InteractionFeatureJob.java:93,113`）與 EXACTLY_ONCE 的 KafkaSink 都沒有設定 `.uid()`。
- stateful ID manifest 只鎖定了 5 個 operator，測試也只比對常數（`InteractionFeatureJobTest.java:505-519`），沒有驗證實際的 JobGraph。
- transactional id prefix 每次都用 `UUID.randomUUID()` 產生（`:428`）。
- 整個專案都沒有設定 `isolation.level=read_committed`。

**影響。**

- 拓撲一變更，不允許 non-restored state 的 restore 就會失敗，最後只能用空 state 啟動，觸發 25。
- 下游讀得到 aborted transaction 的資料。
- 重新提交後，舊 prefix 留下的 transaction 無法被 abort。

**建議與驗收。** 所有 source／sink 都設定 uid，並以實際的 JobGraph 驗證 manifest；固定 transactional prefix，下游 consumer 改用 read_committed。如果做不到，就在文件中明確把語義降為 at-least-once。規模 M；信心 高（read_committed 部分為中）。

### 34．P2：feature_definition_version 無法演進，DIN 經 PIT 拿不到任何資料

**證據。** materializer 只接受 `ranking_ltr_v1`（`FeatureHistoryMaterializerJob.java:440,533`），serving 也永遠送 v1（`recommendation/api.py:1717`）。PIT 會依設定的版本過濾 observation（`PointInTimeFeatureJoinJob.java:456-466`），而 DIN 訓練要求的版本是 `ranking_ltr_v2_din`（`model_trainer/main.py:240-251`）。

**影響。** DIN 走 PIT 時永遠停在 `waiting_for_eligible_rows`。此外，任何特徵版本的升級都會讓事件整批進入 DLQ。

**建議。** 原始歷史固定使用 v1 契約，模型輸入的版本另外標記；materializer 改用版本 allowlist；補一個端到端測試，確認 v2_din 的 PIT 不為空。規模 S–M；信心 中高。

### 35．P3：DLQ 與跨語言 canonical JSON 的小缺口

- **DLQ headers 遺失**：Python DLQ 把 headers 存成 dict（`kafka_client.py:115-125`），但 `scripts/replay_kafka_dlq.py:91` 轉成 list 時全部丟失，與 runbook 寫的「保留 headers」不符。Flink DLQ 則沒有記錄 key／partition／offset，replay 之後就失去依 user 分區的特性。
- **canonical JSON 不一致**：遇到控制字元時，Python 輸出 `\u001b`，Java 輸出 `\u001B`；JDK17 的 `Double.toString` 在少數數值上也不是最短表示（以 Jackson 2.15.3＋JDK17 重現；20 萬個常見亂數浮點數則沒有出現不一致）。兩邊不一致時 payload_hash 對不上，事件就會進入 DLQ。
- **反序列化失敗不進 DLQ**：Python consumer 反序列化失敗時，`consume()` 直接結束（`kafka_client.py:476`），該訊息不會進入 DLQ。

規模 S。

### 3.3 資料管線（離線／PIT）

### 36．P1：預設 `RANKING_TRIMODAL_SHADOW=true`，讓文件上的 PIT preset 每一輪都 fail-closed

**證據。**

- `RANKING_TRIMODAL_SHADOW` 預設為 true（`config.py:1144`、`docker-compose.yml:159`、`values.yaml:363`）。
- `model_trainer/main.py:206-209` 只要 trimodal_shadow 為真，就要求 pinned candidate sidecar。
- `deploy/compose/pit-production.conf:25-28` 的 sidecar 欄位卻全部是空的，reader 因此在 `pit_training_dataset.py:597` 以「manifest is missing candidate sidecar」fail-closed。
- legacy 模式下，`_train_pit_shadow_model` 以 `ranking_pit_dataset_uri=None` 呼叫 `read()`（`main.py:1221,1362`），要到 artifact 已經持久化之後才報錯。

**影響。**

- 照文件啟用 pit-production preset 後，ranking PIT 永遠不會完成訓練，每小時重試一次並觸發 critical alert。
- legacy 模式下，每一輪的狀態都是 error。
- 即使補上 sidecar，shadow 仍走全量記憶體路徑，抵銷了 18 的成果。

**建議與驗收。** preset 明確設定 `RANKING_TRIMODAL_SHADOW=false`；或讓 shadow 改用 streaming source，且只在 sidecar 存在時才啟用。新增一個 smoke test：用 preset 的環境變數建立 trainer，餵入假 manifest。規模 S；信心 高（已讀原始碼確認）。

### 37．P1：PIT 歸因以 user×product 為單位，且沒有過濾 ranker-rejected

**證據。**

- ranking PIT 以 `ON i.user_id=o.user_id AND i.product_id=o.product_id` 做歸因（`PointInTimeFeatureJoinJob.java:477`），沒有用到 impression_id。
- `eligible_observations` 不過濾 event_type（`:456-458`）。
- ranker-rejected 候選寫入同一張表，而且沒有 `featureBundleHash`（`FeatureHistoryMaterializerJob.java:359-399`），但 reader 要求 hash 必須是 64 字元（`pit_training_dataset.py:1018`）。
- 相對地，legacy 路徑是以 impression 為單位歸因（`system_store.py:370-388`）。

**影響。**

- 重複歸因：同一商品在 168 小時內多次曝光時，一次購買就會把所有 observation 都標為正例，value 也被重複計算。organic 點擊則被歸功給推薦。
- rejected 候選的處理：從未顯示過的 rejected 候選，只要之後被點擊就會入選（`:344`）。它的 null hash 會讓整批 PIT 失敗；若放寬檢查，則改為污染訓練資料。
- 標籤跨切分重疊：train 尾端與 holdout 可能共用同一筆 purchase 作為標籤（因為沒有 purge gap）。
- 兩條路徑不一致：legacy 與 PIT 的標籤語義不同，切換時指標無法直接比較。

**建議與驗收。** 改以 `context.impression_id` 做 last-touch 歸因；ranking PIT 限定 `event_type='ranking_observation'`；holdout 之前保留一段長度等於 window＋lateness 的 purge gap。測試：同一商品曝光三次、購買一次，結果只能有 1 筆正例。規模 M；信心：程式路徑高，實際發生頻率中。

### 38．P1：trainer 用來比較的 champion 不是現役 release

**證據。**

- production 必須設定 generation root（`config.py:2864`），現役 checkpoint 安裝在 generation 目錄下（`model_artifacts.py:370-379`）。
- trainer 卻從 `ranking_model_path` 載入 champion（`model_trainer/main.py:381,795-797`）。這個檔案其實是 trainer 自己上一輪存下的 challenger（`:851-855`）；檔案不存在時，`load_model` 會建立一個隨機權重的模型。
- `loaded_ranking_checkpoint` 只在 init 時設定一次。

**影響。** gate 實際比較的是「這一輪 vs 上一輪的 challenger（可能已被否決），或是一個隨機模型」，記錄時用的卻是現役的 `champion_release_id`，因此可能產生虛假的 PASS 並進入 staging。另外，v1 還會從這個檔案 warm-start，導致模型漂移且不可重現。

**建議與驗收。** champion 一律從現役 release 的 `checkpoint_path` 載入，並驗證 sha；challenger 寫到獨立的路徑；activation 時檢查 `champion_release_id == pointer.release_id`。補一個開啟 generation root 的 trainer gate 整合測試。規模 S–M；信心 高。

### 39．P1：每日 PIT 都重算全部歷史，資料與暫存都沒有上限

**證據。**

- `eligible_observations` 只有時間上界（`PointInTimeFeatureJoinJob.java:458`），每個 run 都把全量資料 append 進 `ranking_training_pit`；repo 裡沒有任何刪除邏輯。
- trainer 的 `verified/ranking-<sha>` 永遠不會清除（`pit_streaming.py:203`）。
- 空間上限：Helm emptyDir 的 `sizeLimit` 是 20Gi（`values.yaml:490`），而單一 generation 的上限也是 20GiB；Compose 用的則是持久化 volume。

**影響。** 儲存成本隨天數呈平方成長，Flink 與訓練時間也線性變長，最終會超過 6 小時的 lease。幾天內 pod 就會因 ephemeral-storage 不足被驅逐，或把 Compose 的磁碟寫滿。

**建議。** 改為增量 PIT，並加上訓練 lookback 的設定；為舊 run 訂定保留政策（與 Iceberg lineage 保護一致）；trainer 只保留 current 與 previous 兩個 generation。規模 M；信心 高。

### 40．P1：每日 orchestrator 不確認 materializer 已追上 cutoff

**證據。** materializer 的 readiness barrier 只是 Compose 的一次性依賴（`docker-compose.yml:1292-1295`）；`run_once` 與 Helm CronJob（`--once`）都不檢查 materializer 是否已追上 cutoff。lateness 設為 1 小時（`config.py:2086`），而 available_at 用的是處理時間（`InteractionFeatureJob.java:700`）。

**影響。** 若 02:00 時 materializer 落後超過 1 小時，成熟樣本中的正例會被靜默遺失，而 manifest 是不可變的。晚到的事件會出現在下一個 run，同一筆 observation 的標籤因此跨 run 改變。

**建議與驗收。** run 之前先檢查 consumer offset 與 checkpoint 時間，尚未追上就標記為 waiting；量測 label 延遲的分佈，再據以調整 lateness。規模 S–M；信心 中高。

### 41．P1：PIT 品質 gate 大多只是裝飾

**證據。**

- `model_trainer/main.py:855-866` 把 assembler parity、label reconciliation、current_state_calls、invalid_rows 分別寫死為 1.0、1.0、0、0。
- `pit_online_offline_parity_ratio` 與 `pit_leakage_rows` 只有定義（`observability.py:766-772`），卻掛上了 critical alert（`prometheus-rules.yml:426,471`）。
- manifest 裡的 parity 與 quarantine 欄位（`pit_manifest.py:226-227`）沒有任何程式檢查。
- `evaluate_shadow_gate`（`feature_lake_shadow_gate.py:82`）沒有任何呼叫者。

**影響。** parity 只有 50%、或 quarantine 高達 90% 的資料集照樣會被拿去訓練，相關 alert 也永遠不會觸發。監控因此給出虛假的安全感。

**建議與驗收。** 在 reader 端依門檻 fail-closed，涵蓋：parity、quarantine 比例、row 數與 label rate 相對上一個 run 的變化、null rate。gauge 改為回報真實量測值；並加一個測試，確認低 parity 的 manifest 會被拒絕。規模 S–M；信心 高（已讀原始碼確認）。

### 42．P1：Retrieval PIT 的 Java／Python 契約漂移

**證據。**

- `RetrievalPointInTimeJoinJob.java:125` 寫死 `'[]'` 與 `CAST(NULL AS DOUBLE)`，所以每一列都是 `seen_product_ids=[]`、`ranker_score=NULL`。
- Java 的 organic positive 只要求 `event_time <= cutoff`（`:93-95`），Python 卻對每一列都強制成熟條件（`retrieval_pit_dataset.py:678-687`）。
- Java 測試只比對 SQL 字串。

**影響。**

- 評估的 `eligible - seen` 永遠不會排除已互動過的商品，但 serving 會排除 `exclude_items`，所以離線指標會獎勵「把已買過的商品推回去」的模型。
- teacher_soft 的目標恆為 sigmoid(0)=0.5（`retrieval_training.py:235`）。這組實驗沒有任何資訊量，卻被當成一個負樣本版本來比較。
- 只要近 7 天內有 organic 點擊，整批資料就會驗證失敗；若放寬檢查，holdout 又只剩 organic 正例，而且可能洩漏。

**建議與驗收。**

- 依 as_of 從 interaction history 產生 seen。
- ranker_rejected 帶入真實的 logit，並標明分數的語義。
- organic 列套用成熟條件，或另外切成獨立的 split。
- 以 MiniCluster 或 Python 參考實作做語義測試；manifest 驗證 teacher_soft 時 ranker_score 的非空比例。

規模 M；信心 高。

### 43．P1：Retrieval 評估是 O(Q×N)，還把完整排序清單寫進 Postgres

**證據。**

- 重複計算：`retrieval_training.py:97-150` 對每個 query 重算全 catalog 的 `cold_items` frozenset 與 `missing_by_name`；`_popularity_scores`（`retrieval_release_training.py:312-333`）為每個 query 建一份全 catalog 的 dict。
- 計算方式：exact scoring 是純 Python 逐 item 呼叫 `np.dot`（`retrieval_evaluation.py:80-104`）；bootstrap 跑 2,000 次，每次對每個 query 都重建 `set(eligible)`。
- 持久化：完整的 `ranked_product_ids`（`retrieval_evaluation.py:224`）被寫進 metrics（`retrieval_release_training.py:197`）。
- 比較不對稱：challenger 先截到 top-200 再扣除 seen，popularity 卻保留全部 N 筆；gate 看的是 Recall@100/200，但 serving 的 Two-Tower 只取 40 筆（`config.py:404`）。

**影響。**

- 1k query × 50k 商品就有約 5,000 萬個字串，記憶體會爆，也會超過 JSONB 上限。此時 release 已經 register，evaluation 卻寫不進去。
- 修好 42 之後，截斷方式的不對稱會變成不公平的比較；gate 的 cutoff 也與 serving budget 不一致。

**建議與驗收。**

- 與 query 無關的集合只計算一次。
- per-query 指標只算一次，bootstrap 只重抽 index。
- exact scoring 改用矩陣運算或分塊的 `IndexFlatIP`。
- popularity 只保留 top-(K+|seen|)；持久化前移除排序清單。
- 新增 Recall@40，對齊 serving budget。

驗收：N=10 萬、Q=1 萬時，能在固定的 RSS 與時間內跑完。規模 M；信心 高。

### 44．P2：離線管線的其他缺口

- **lineage 保護對 release 實際無效。** `collect_snapshot_references` 需要同一個 dict 裡同時有 `iceberg_table_id` 與 `iceberg_snapshot_id`（`iceberg_maintenance.py:33-38`），但 ranking 的 bundle manifest 沒有這兩個鍵（`model_artifacts.py:1234-1275`）。目前只保護 30 天內完成的 ranking run（`system_store.py:4350-4367`）；retrieval PIT 與來源表的 snapshot 都不受保護，export prefix 也沒有保留政策。
- **失敗重試沒有退避。** 失敗的 PIT 訓練可以立即被重新 claim（`system_store.py:3945-3962`），每小時重新下載資料並重建 spool，沒有次數上限或退避。確定性的失敗應標記為 terminal。
- **observation_id 沒有唯一性約束。** spool 沒有對 observation_id 設 UNIQUE，重複時要到 `RankingEvaluationSpool` 的 PK 才會丟出 IntegrityError（`model_release_quality.py:83`）。
- **設定寫死與 lineage 欄位缺漏。** Phase C 把 `holdout_days=7` 與 `calibration_fraction=0.1` 寫死（`train_phase_c.py:66-70`），可能與 trainer 的設定不一致；streaming 模式的 lineage 少了 `quarantine_rows`。

規模 S–M。

### 3.4 DL 模型：排序

### 45．P1：離線 NDCG 被沒有點擊的 slate 稀釋，門檻失真

**證據。** `ranking_training.py:160-165` 把 view 的 relevance 設為 1.0，`_mean_ndcg`（`model_release_quality.py:653-680`）也沒有跳過沒有正例的 slate。當一個 slate 內所有 relevance 都相同時，`dcg(ranked)==dcg(ideal)`，NDCG 恆為 1。

**影響。** 如果只有 3–5% 的 slate 有點擊，量到的 Δ 就只剩真實 Δ 的 3–5%。features 實驗要求 Δ≥0.005（`model_release_quality.py:412`），換算成真實提升約需 0.1；而 CI 下限 −0.005 卻會放行約 −0.1 的真實退步。模擬 20% 的 slate 有點擊時，隨機排序的 challenger 的 NDCG 只從 1.0 降到 0.9807。

**建議與驗收。** view 的 gain 改為 0（或改用 2^rel−1 並讓 view=0）；沒有正例的 slate 不納入 NDCG，並回報 informative slate 的數量；加入 per-impression／per-user GAUC。補一個全 view slate 的單元測試，再重新校定門檻。規模 S；信心 高（已讀原始碼確認）。

### 46．P1：任一小切片就讓整個 gate 變成 INSUFFICIENT，且門檻無法調整

**證據。** 每個 category、retrieval source、missing modality 都會成為一個切片（`model_release_quality.py:534-560`），只要有一個切片證據不足，整體就判為 `INSUFFICIENT_EVIDENCE`（`:385-391,405`）。`quality_gate_snapshot_valid` 要求設定必須等於預設值，因此也無法放寬。實測：兩個相同的模型，加上一個只有 1 次曝光的類別，結果是 `insufficient_evidence ['slice_category_tiny']`。

**影響。** 在真實的長尾類目下，gate 永遠過不了。另外，數十個切片各自使用點估計門檻（例如 ECE ±0.01、log loss 5%），多重比較會造成大量的偽 FAIL。

**建議與驗收。** 只有預先登記的關鍵切片（cold user／item、價格帶）證據不足時才阻擋；類目只取曝光量 Top-N 再加一個 "other"，其餘只報告、不阻擋；切片指標改做 bootstrap CI，或套用 Holm／BH 校正。驗收：長尾類目的 fixture 可以 PASS；A/A 測試的偽失敗率低於 5%。規模 S–M；信心 高。

### 47．P1：ListNet 作用在幣值尺度的 canonical score 上，污染 value head 與校準

**證據。** `ranking.py:4091-4130` 用 `canonical_score_torch(ctcvr, inverse_transform(gmv))` 計算 LTR，而且沒有 detach。listwise 預設開啟（Compose 中 weight 為 0.1），value loss 則只看購買列（`:4205-4232`）。實測顯示：`value_mask=0` 的列，gmv 梯度約為 0.0027，方向是把 value 調低；ctr 的梯度則被 value 放大。

**影響。**

- 非購買列的 value head 唯一的訓練訊號來自 ListNet，因此會學成「購買傾向越高，value 越高」，在 ctcvr×value 中等於重複計算。value WAPE 只在購買列計算，所以 gate 看不到這個問題。
- score 的尺度是金額：高價 slate 的 softmax 很尖，低價 slate 幾乎是均勻分布。LTR 實際上只在訓練高價商品，CTCVR 的校準也被扭曲。

**建議與驗收。** 改用 `log ctcvr + log value.detach()`、對 value 加 stop-grad，或以 slate 為單位正規化。消融比較 listwise weight 0 與 0.1，並比較同一商品在購買列與非購買列上的 predicted_value 分布。規模 M；信心 中。

### 48．P1：完全沒有 position／exposure bias 處理

**證據。** impression 有儲存 position（`system_store.py:1848`、`recommendation/api.py:983`），但 PIT example 沒有帶入（`pit_training_dataset.py:1120-1140`）。特徵中只有 `session_position`（`ranking.py:738-760`），gate 也沒有任何 IPS 類的指標。

**影響。** CTR 學到的是 champion 排序所造成的位置效應，形成 feedback loop。logged-slate 上的 NDCG 天生偏袒「排序方式像 champion」的模型，會讓曝光新商品的 challenger 吃虧。

**建議與驗收。** 訓練時加一個 shallow position tower（additive logit，serving 時固定在參考位置），或採用 PAL；以少量 top-k swap 隨機化來估計 propensity；gate 加入 SNIPS-NDCG。移除 tower 之後，ONNX parity 仍須通過。規模 M；信心：缺失本身高，影響幅度中。

### 49．P2（啟用 DIN 前必修）：DIN 的 Dice 在 eval 時仍用 batch 統計，分數會隨同批的請求改變

**證據。** `din.py:404` 的 Dice 直接用當前 batch 的 mean/var，沒有 running stats，而且統計量還包含被 mask 掉的 padding 位置。`ranking_batcher.py:241-288` 會把多個請求合成一次 forward。實測：同樣 4 列單獨跑，與和 16 列合批跑，最大差異 0.026（輸出平均絕對值為 0.40，約 6%）。ONNX parity 是在相同的 batch 上比對（`ranking_onnx.py:189-223`），所以抓不到。

**影響。** 啟用 DIN 時，排序不可重現、cache 也不一致。訓練 batch 是 1024，serving 卻常常只有單一請求，形成 train/serve skew；Triton dynamic batching 也有同樣的問題。

**建議與驗收。** Dice 改用 BN 式的 running stats（只統計有效位置），或改用 PReLU；加一個 batch-invariance 測試，要求同一請求在合批前後的分數差小於 1e-5。規模 S；信心 高。

### 50．P2／P3：排序訓練與評估的其他缺口

- **P2｜gate bootstrap 無法擴展到真實資料量。** 每個 sample 都重建每一列的 dataclass（`model_release_quality.py:683-712`），`spool.load()` 會把全部資料載入記憶體（`:113-140`）。實測每列每 sample 約 3.65µs：1M 列 × 2,000 次光是 overall 就約 2 小時，再加上切片約 20 小時。建議改為每個 impression 先算一次 NDCG，彙總成 per-user 的 sum/count，再用 numpy multinomial 權重做向量化 bootstrap。驗收：1M 列在 5 分鐘內完成。
- **P3｜數值穩定性。** 先做 sigmoid 再用 `nn.BCELoss`（`ranking.py:4056,4164-4203`），應改為 BCEWithLogits，並讓 ctcvr 在 log 空間相乘。
- **P3｜驗證指標只代表單一 batch。** `metrics_tensors` 只保留最後一個 validation batch（`ranking.py:2643,2683`），所以寫進 release payload 的 auc／ndcg 只反映單一 batch。
- **P3｜訓練可重現性與調參。** v1 路徑沒有設定 `torch.manual_seed`（只有 v3 在 `:2564` 設定）；沒有 LR schedule 也沒有 HPO；Adam 的 weight_decay 是 L2 正則，而非 decoupled（AdamW）。

規模 S。

### 3.5 DL 模型：召回與多模態

### 51．P1：production 沒有 SASRec／Swing／cluster／trending／冷啟 adapter 的產生路徑

**證據。**

- 這些來源都只在 `_update_models_from_interactions`（`recommender.py:1744-1823`）中產生。該路徑受 `online_retraining_enabled` 控制（`:3093`），而 production 強制為 false（`config.py:2884`）；PIT 模式的 trainer 也不會建立 RecommendationEngine（`model_trainer/main.py:333`）。
- `trending_scores` 只在 `recommender.py:1813` 更新，所以 production 的 trending pool 實際上只是 rating／review／freshness 的靜態分數（`:2119-2142`）。
- PIT 的 `write_two_tower_bundle` 不產生冷啟 adapter，`_load_cold_start_sidecars` 因此被清空（`:448-458`）。
- SASRec／Swing 透過 `sync_latest_*` 載入，繞過了 release gate（`:1619,1647`）。

**影響。** production 即使打開這些 flag 也拿不到 artifact；若有人手動產出，最新的 artifact 會不經品質門檻直接上線。「trending」這個來源名不符實，也會影響 RRF 與來源分析。

**建議。** 為這些來源建立 offline／PIT job 與 release registry，並沿用 Two-Tower 的 gate；trending 改用 Flink 視窗計數；readiness 回報各來源 artifact 的年齡。規模 M–L；信心 高。

### 52．P2：new-item 刷新在 serving 程序內是 O(N²)

**證據。** `cf_cold_start.py:290` 每次呼叫 `if product_id in set(trained_item_ids)` 都會重建一次 set；`recommender.py:566-585` 對全 catalog 逐一呼叫，並每 300 秒以 `to_thread` 執行一次（`config.py:512`）。PIT 模式下 item mapping 等於整個 eligible catalog（`two_tower.py:1208-1210`）。另外，`_get_clip_neighbors` 為每個冷商品複製全部 embedding 再暴力掃描（`cf_cold_start.py:679-709`）。

**影響。** N=10 萬時，每輪約有 10¹⁰ 次元素複製；這段計算持有 GIL，會與 serving 爭搶 CPU，造成週期性的 p99 尖峰。

**建議。** set 只建一次；鄰居查詢改用 FAISS；加上刷新耗時的 metric。規模 S；信心 高。

### 53．P1：只部署 visual bundle 時，content-worker 在 required 模式下啟動失敗

**證據。** `video_processor.py:123` 一律呼叫 `load_index()`，而 required 模式缺檔時會直接 raise（`vector_search.py:155-159`）。Helm 的 `/app/models` 是 emptyDir（`backend-deployments.yaml:181`），沒有舊 index 的同步機制；worker 實際上只用到 no-op 的 `add_content_embedding`。文件說 visual bundle 不需要舊檔（`event-and-artifact-reliability.md:45-49`），但這只對 recommendation 成立。model-trainer 的 trimodal shadow 也有相同的依賴（`model_trainer/main.py:329-331`）。

**影響。** 只部署真實 visual bundle 的 K8s production 環境，content-worker 會 crashloop（再加上 68 的問題）。

**建議。** worker 不載入商品 index，或改為 optional；trainer 比照處理。規模 S；信心 中高。

### 54．P1：CLIP／OCR 失敗時靜默產生零向量，並標記為 completed

**證據。** `content_processor.py:1261-1266` 在 CLIP 發生例外時回傳 `torch.zeros`，`process_video` 繼續往下執行，worker 照樣送出 `has_visual_embedding=True`。OCR 發生例外時則回傳空結果（`:1417`）。

**影響。** CUDA OOM、模型檔損毀這類錯誤，會繞過 02 剛修好的 retry／DLQ。零向量被寫進 Redis、DB 與 cluster：在 IP index 下 content 召回為空，在 L2 下則是任意結果，而 job 仍顯示 completed。

**建議與驗收。** 主模態失敗時直接 raise（可重試）；次模態失敗則以 `modality_status` 記錄；監控零範數向量的比例。測試：模擬 CLIP 例外時，job 會進入 retry，而不是 completed。規模 S；信心 高。

### 55．P2：召回模型的其他缺口

- **快照之後才出現的新用戶沒有個人化。**
  - 現況：`two_tower.py:1732` 以 `user_mapping.get(user_id, 0)` 取 ID；`_init_weights` 用 normal_ 覆寫了 padding row（`:416`），而 padding row 不會被更新，所以所有快照外的用戶都共用同一個隨機向量。`UserFeatureEncoder` 只有 10 個聚合值，類別只取長度（`:282-319`），也沒有序列特徵。
  - 影響：release 是手動 CAS，週期越長，落入 0 號的活躍用戶就越多。評估的 `cold_user` 以 `total_interactions<5` 定義，量不到「不在 mapping 裡」的這群人。
  - 建議：訓練時做 ID dropout 或 hash bucket；以近期 positives 的 item embedding 做 pooling；新增 not-in-mapping 切片。新商品則在 catalog 啟用時，增量跑 item tower 補進 ANN。
- **外部負樣本永遠取同一批最舊的。** `negatives_for_user`（`two_tower.py:192-214`）沒有 ORDER BY 也沒有抽樣，走索引所以是升冪；sampler 依序取前 3 筆 impression 與 1 筆 ranker_rejected（`:777-799`）。另外，每個 batch 用 O(B²) 的 Python 迴圈建立 in-batch mask（`:1456-1464`）。建議以 (seed, epoch, sample) 做決定性抽樣，或取最接近 as_of 的視窗；mask 改為向量化計算。
- **v2 mean 視覺向量沒有 CLIP lineage。** 只有 v3 的 `retrieval_clip_*` 帶有 lineage（`models.py:455-458`），fallback 路徑（`visual_retrieval.py:706-719`）直接拿 `visual_embedding` 去查任何 product index。CLIP revision 升級後，舊的內容特徵會與新 index 混用，而且不會報錯。
- **SASRec／Swing 的擴充性。** `sasrec.py:289` 每個請求都重建 `set(catalog_product_ids)`，`:323` 掃過整個 vocab。`swing_itemcf.py:164` 以 `sorted(users)[:500]` 依字典序截斷，而不是隨機或取近期；`:170` 的全域 `user_pairs` 在 1 萬個用戶時可達約 5×10⁷ 個 tuple。

規模 M。

### 56．P2：中文影片文字大多進不了特徵

**證據。** OCR 預設語言為 `en`（`config.py:183`、`docker-compose.yml:630`）。Tesseract 參數為 `--oem 3 --psm 6`，沒有加 `-l chi_tra`（`content_processor.py:70`）。`len(track.text) > 2` 會刪掉「手機」「耳機」這類兩字詞（`:1395`）。價格偵測不認得「元」或「NT$」（`:1434-1450`）。然而類別關鍵字表本身卻含有繁簡中文（`:117-192`）。

**影響。** 中文影片的字幕與商品文字大多進不了 OCR 與 text embedding，多模態特徵在主要市場上可能幾乎缺失。

**建議與驗收。** 依市場把 PaddleOCR 改為 `ch` 或 `chinese_cht`；長度門檻改依 CJK 字元判斷；補上幣別的 regex。驗收時用一組中英字幕影片，比較 OCR 覆蓋率與下游 Recall。規模 S；信心：程式碼面高，實際影響中。

### 57．P3：多模態處理的其他改善

- `_detect_objects` 對每支影片又拿 3 張 frame 跑一次 CLIP，再加上 30 個 text prompt（`content_processor.py:1452-1520`）；其實可以重用已經算好的 frame embedding。
- E5 的 encode 沒有分批（`text_embeddings.py:52-60`），OCR／ASR 段落多時，padding 會造成記憶體尖峰。

規模 S。

### 3.6 線上服務與可靠性

### 58．P1：遠端排序拓撲下，recommendation-service 永遠不會熱更新 retrieval artifact，版本 context 也凍結

**證據。**

- `_periodic_ranking_checkpoint_sync` 只有在 `ranking_model is not None` 時才會建立（`recommendation/api.py:1328-1335`）。
- 預設拓撲是遠端 coordinator，且 `SERVICE_RANKING_SERVICE_FALLBACK_ENABLED=false`（`config.py:2379`、`docker-compose.yml:260`），因此 `_should_initialize_local_ranker`（`:365-368`）回傳 False，ranking_model 為 None。
- `sync_serving_artifacts_if_updated()` 與 `_refresh_serving_version_context()` 唯一的呼叫點，就在這個 task 裡面（`:3252-3256`）。

**影響。**

- 新的 Two-Tower generation、content cluster 等 retrieval artifact，都要等 pod 重啟才會生效，rollout observation 也不會回報。也就是說，PROGRESS 記錄的 retrieval 熱切換，在預設拓撲下其實沒有接上。
- cache key 的 `serving_versions` 不包含遠端 ranker 的版本。ranking release 之後，最多 900 秒內仍會回傳舊模型產生的 slate；impression lineage 也缺少 `model_release_id`。

**建議與驗收。** artifact 同步迴圈不再依附於 ranking_model；遠端模式改由 coordinator 回報的 profile 取得 ranker 版本，並納入 cache key 與 lineage。用遠端拓撲做整合測試：promote 一次之後，recommendation 在一個 poll 週期內切換到新的 generation。規模 S；信心 高（已讀原始碼確認）。

### 59．P1：readiness 級聯，任一依賴失敗就讓整站下線（包含靜態前端）

**證據。**

- `service_common.py:222` 的判斷是 `ready = all(status == "healthy")`，degraded 也算失敗。
- 探測鏈：gateway `/readyz` 依序探測 recommendation、ranking、ingest、Kafka、DB、S3（`gateway/api.py:290-345`）；recommendation `/readyz` 又要求 Kafka、DB 與 ranking 都是 healthy（`recommendation/api.py:1435-1483`）。Caddy 的 readinessProbe 打 `/readyz`，請求會被轉到 gateway（`caddy-deployment.yaml:63-66`）。
- Kafka 初始化：失敗時，三個服務都把 `kafka_manager` 設為 None 且永不重試（gateway `:209`、ingest `:82`、recommendation `:1345`）；liveness 只看 `/livez`。
- 公開端點：`/readyz` 與 `/health` 列在 `_AUTH_EXEMPT_PATHS` 中（`gateway/api.py:67`），不驗證、不限流，並經由 Caddy 對外公開。

**影響。**

- 任一依賴故障（例如單一 coordinator、Kafka，或 S3 health 抖動），都會讓所有 gateway pod 變成 NotReady，接著所有 Caddy pod 也 NotReady，結果連靜態 SPA 和原本還能用的 ingest 都一起回 503。
- 如果 Kafka 在 pod 啟動時短暫不可用，pod 會永久 unready，而且不會被重啟。
- 公網每打一次 `/readyz`，就被放大成 DB 查詢、S3 list、三個 HTTP probe，以及對所有 runner 的 health 檢查；回應中還會帶出內部錯誤字串。

**建議與驗收。**

- Caddy 只對自身做 probe。
- 各服務的 readiness 只檢查自身必要的依賴；下游失敗時回 degraded，並交由 alert 處理。深度狀態移到受保護的 `/readyz/full`。
- Kafka 改為背景重連。
- 對外的 `/readyz` 移除，或加上短 TTL 的 cache。

驗收：用 kind 停掉 ranking 或 Kafka，確認首頁仍可用、Caddy 與 gateway 仍維持 Ready、`/api/interactions` 回 503。規模 S–M；信心 高。

### 60．P1：Coordinator／Runner 的 TCP 協定沒有認證，DRAIN 不可逆

**證據。** runner 收到 `DRAIN_OPERATION` 就執行 `_handle_drain`（`ranking_runner/main.py:280`），`:538` 設定 `_draining=True` 後沒有任何解除路徑，`:502` 的 readiness 因此永久為 false。coordinator 與 runner 都 bind 在 0.0.0.0，也不檢查 internal key。Helm 只有 Triton 有 NetworkPolicy（`ranking-triton.yaml:159-162`）。

**影響。** 叢集內任一 pod（或透過 SSRF）只要送出一個 4-byte 的 frame，就能讓 runner 永久 unready；而 liveness 只做 TCP 檢查，不會重啟它。此外，RANK 走 TCP，也繞過了 HTTP 層的 internal key 檢查。

**建議與驗收。** 連線時先做 HMAC／共享金鑰握手，或改用 mTLS；為 coordinator 與 runner 補上 NetworkPolicy；drain 只接受 loopback，並提供 undrain。測試：未授權的 frame 會被拒絕。規模 S–M；信心 高。

### 61．P1：多 uvicorn worker 下，Prometheus 指標只代表單一 process

**證據。** 每個 process 有自己的 `CollectorRegistry`（`observability.py:169`），`/metrics` 以 `generate_latest(self.registry)` 輸出（`:1666-1667`）；repo 中沒有 multiprocess 模式，也沒有 `PROMETHEUS_MULTIPROC_DIR`。gateway 跑 `--workers 4`（`docker-compose.yml:373`、`values.yaml:498`），recommendation 在 Compose 是 8 個、在 Helm 是 4 個，ranking 與 ingest 也都是多 worker。

**影響。** 每次 scrape 隨機拿到其中一個 worker 的值：counter 看起來不斷 reset，`rate()` 被高估，histogram 的 p95／p99 也會跳動。overview dashboard、SLO，以及 latency／5xx 告警（例如 `Ranking5xxDetected`）都不可信。先前的負載測試結論如果依賴這些指標，也需要重新核對。

**建議與驗收。** 啟用 prometheus_client 的 multiprocess mode，或改成每個 pod 只跑單一 worker、以增加副本數來擴展。驗收：在固定流量下對同一 pod 連續 scrape，counter 單調遞增。規模 S–M；信心 高。

### 62．P1：Postgres 在熱路徑上沒有 timeout，連線預算也會爆

**證據。**

- 沒有 timeout：`create_async_engine(pool_size, max_overflow, pool_pre_ping)`（`system_store.py:1364-1370`）沒有設定 `pool_timeout`、`command_timeout` 或 `statement_timeout`，SQLAlchemy 預設會等待連線 30 秒。
- 熱路徑上的寫入：recommendation 在每個請求中（包括 cache hit 與 singleflight joiner）都直接 `await enqueue_recommendation_event`（`recommendation/api.py:3063`），外層沒有 `wait_for`；ingest 每筆互動要跑 2 個 transaction。
- 連線預算：每個 process 有 5＋5 條連線（`values.yaml:117`），共有 14 個服務會建立 SystemStore，runner 最多 32 個副本。在 HPA 上限時約需 1,200 條以上的連線，而且沒有 PgBouncer。

**影響。** DB 一變慢，連 cache hit 都會卡住 30 秒，接著 gateway 回 504。HPA 擴容時會撞上 `too many connections`，ingest ledger 與 outbox 同時失效。

**建議與驗收。**

- 設定 statement_timeout 與 pool_timeout。
- outbox 寫入加上時間預算（例如 50ms），超時就回傳 `impression_tracking: unavailable` 並記錄 metric。這需要與 05「先持久化才回傳 impression_id」的契約一併決定。
- 加上 PgBouncer，並建立一張連線預算表。

驗收：注入 DB 延遲後，推薦的 p99 不超過預算。規模 M；信心 高。

### 63．P2：gateway 的 rate limiter 實際上是全域共用一個桶，Redis 一故障整站就回 500

**證據。** `gateway/api.py:153` 用的是 `client_id = request.client.host`。uvicorn 沒有加 `--forwarded-allow-ips`（`docker-compose.yml:373`），Caddy 帶來的 X-Forwarded-For 因此被忽略，所有請求都以 Caddy 容器的 IP 計數，上限為 120,000/min（`config.py:1686`）。`service_common.py:68` 的 `await pipe.execute()` 沒有例外處理，而 Redis 的 `socket_timeout=30`（`config.py:47`）。限流排在認證之後。

**影響。** 單一濫用者就能吃光全站的額度，讓所有人一起收到 429。Redis 卡住時，每個請求都要等 30 秒才回 500，連不需要 Redis 的路徑也一樣。

**建議。** 信任 Caddy 的網段並使用 XFF，或改以 API key／subject 分桶；限流改為 fail-open，搭配短 timeout 與 metric；未認證的請求也要限流。規模 S；信心 高。

### 64．P2：Ingress 與上傳沒有請求大小限制

**證據。**

- 上傳：使用 `UploadFile = File(...)` 時，FastAPI 會在進入 handler 之前先把整個 multipart spool 到 /tmp，`max_file_size` 要到 staging 階段才檢查（`gateway/api.py:411-415,733`）。
- 缺少上限設定：Caddyfile 沒有 `request_body max_size`；Helm 的 `emptyDirs.tmp: {}` 沒有 sizeLimit（`values.yaml:488`）。
- 無作用的設定：`api_config.max_request_size` 與 `max_concurrent_requests` 沒有任何程式讀取（`config.py:1674`）。
- JSON payload：`context: Dict[str, Any]` 沒有大小或深度限制，會原封不動寫進 outbox 與 Kafka（連動 05 提到的 1MB 問題）。

**影響。** 大量的多 GB 上傳會在被拒絕之前耗盡節點的 ephemeral storage，並觸發 eviction。

**建議。** Caddy 依路由設定 `max_size`；tmp 加上 sizeLimit／ephemeral-storage limit；限制 context 的大小與深度；移除無作用的設定。規模 S；信心 高。

### 65．P2：線上服務的其他缺口

- **Singleflight 把 owner 的 499 廣播給 joiner。** owner 偵測到斷線後會 `raise HTTPException(499)`（`recommendation/api.py:2095`），這個例外被設到共享的 future 上（`:2336-2342`），joiner 以 `await asyncio.shield(future)` 拿到的就是同一個例外（`:1777`）。client 逾時重送時，如果原連線剛好斷開，重送的請求必定失敗。建議 owner 斷線時繼續算完，或改由 joiner 接手；joiner 也應加上 timeout。
- **Recommendation cache key 缺少 Phase C 的 categorical context。** `ranking_categorical.py:215-221` 會讀取 surface／placement、time_of_day、device_type、request_category，但 `_build_recommendation_cache_context`（`recommendation/api.py:255-272`）只包含 device、page、category、session_position、time_on_page、hour。v3 promote 之後，不同的 surface 會共用同一份 slate，記錄下來的 feature_context 與實際排序依據不一致。建議從 feature contract 推導 cache 欄位，並加上 parity test。
- **TCP client 在取消時不安全。** `ranking_coordinator_client.py:138` 只在 `except Exception` 時才 close 連線，CancelledError 發生後，`:237` 仍會執行 `put_nowait(connection)`；`ranking_runner_client.py:104-125,711-725` 是同一個模式。被取消的連線帶著未讀的回應回到池裡，下一個請求就可能讀到上一個請求的回應，造成跨使用者的錯配。建議改為 `except BaseException` 時 close，並比對回應的 request_id（信心 中）。
- **商品 metadata 記憶體快取永不失效，而且 fail-open。** `feature_store.py:113,1537-1555` 是一個沒有 TTL 的 dict，啟動時以 vector catalog 預先填入；而 `_is_recommendable_product(None)` 會回傳 True（`recommendation/api.py:799-808`）。結果是下架或缺貨的商品要等 pod 重啟才會消失，metadata 已被移除的商品也仍可能從凍結的 Two-Tower index 被推薦出去。建議加上 TTL，或由 catalog activation 事件觸發失效；production 缺少 metadata 時應 fail-closed。
- **Bearer 驗證在 event loop 內同步抓取 JWKS。** `auth.py:75-79` 的 `PyJWKClient.get_signing_key_from_jwt` 使用同步的 urllib（預設 timeout 30 秒），遇到未知的 kid 會強制 refresh；production 的檢查（`config.py:2999-3032`）也不要求 audience 與 issuer。持續送入隨機 kid 的 token，就能讓 gateway worker 一直阻塞；同一個 IdP 簽給其他 app 的 token 也會被接受。建議 JWKS 改為非同步抓取並快取，對 refresh 做限流；強制檢查 aud、iss，並設定 `require=["exp"]`。

規模 S–M。

### 66．P3：線上服務的小項

- `require_internal_service_auth` 在 middleware 內 raise HTTPException，實際回傳 500 並輸出 traceback（已在主機 FastAPI 上用 probe 確認）。三處 key 比對都沒有使用 `hmac.compare_digest`。
- gateway 的 auth middleware 位於最外層，因此 401／429 不會進入 http metrics，也不會帶 CORS header。
- `_bounded_hot_path_read`（`recommendation/api.py:2959`）timeout 後回傳 fallback（例如預設的 user features），只寫 log、不記 metric。cache race 只等 5ms，Redis 一變慢，命中率就崩落，進而放大 ranking 的負載。
- coordinator client 的 pool wait、write、read 各自有一份 2 秒的 timeout，另外還有一次重試，加總起來超過 body 內的 deadline；而且這個 deadline 是跨主機的 wall clock。
- catalog outbox 失敗時沒有 backoff，也沒有 terminal 狀態（`system_store.py:3611`）。lease 用的是應用端時鐘，而非 DB 的 `now()`。ingest `:297` 以 fire-and-forget 建立的 task 沒有保留引用。

規模 S。

### 3.7 部署、維運、安全與前端

### 67．P1：前端 Live 模式上傳送出的是 JSON，不是 multipart

**證據。** `Code.Frontend/src/api.js:4-10` 在 axios instance 上設定了 `headers: {'Content-Type': 'application/json'}`，而 `uploadVideo` 以 `api.post('/api/content/upload', formData)` 送出（`:95-103`）。axios 1.x 的 `transformRequest` 遇到 JSON content-type 時，會把 FormData 轉成 JSON；以 stub adapter 實測，送出的內容是 `{"file":{}}`。gateway 端要求的是 `file: UploadFile = File(...)`（`gateway/api.py:388`）。

**影響。** 從 UI 走 Live 模式上傳，一定得到 422。vitest 與 e2e（`e2e/workbench.spec.js:5,43`）都只跑 Mock 模式，所以一直沒被發現。

**建議與驗收。** 移除 instance 層的預設 Content-Type，只在 JSON 請求上設定（或在上傳時明確指定 multipart）；補一個 adapter 測試，assert `config.data instanceof FormData`；再加一個 Live 模式的 e2e。規模 S；信心 高。

### 68．P1：Helm 的 content-worker 沒有 PaddleOCR runtime

**證據。**

- Helm 的所有 workload 都用 `video-commerce.backendImage`（`backend-deployments.yaml:61`；`values.yaml:19-22` 只定義了一個 backend image），也就是 Dockerfile 的 production target。
- Paddle venv 只存在於 `content-worker` target（`Dockerfile:133-141`）。
- `ocr_backend` 預設為 `paddle`（`config.py:179-180`），Helm configmap 也沒有設定 `MODEL_OCR_BACKEND`。`video_ocr.py:136` 會 Popen `/opt/paddleocr/bin/python`，失敗時 `content_processor.py:274-295` 會 re-raise。
- Compose 用的是正確的 target（`docker-compose.yml:620`），所以本機看不出問題。`pit-orchestrator` target 也是同樣的情況。

**影響。** 在 K8s 上，content-worker 一初始化模型就失敗，任何影片都處理不了；PIT orchestrator 也可能缺少 Flink job jar。

**建議與驗收。** 新增 `images.contentWorker` 與 `images.pitOrchestrator`（或讓每個 workload 都能指定 image）；CI build 這兩個 target；用 helm template 驗證 content-worker 指向含 Paddle 的 image。規模 S；信心 高（已讀原始碼確認）。

### 69．P1：Alert 送不出去，也沒有 runbook

**證據。** `monitoring/alertmanager.yml` 只有 `receivers: - name: default`，沒有任何 integration。60 條 alert 中，帶 `runbook_url` 的數量為 0。Helm chart 沒有 PrometheusRule template，`slo-and-alerting.md` 也沒說明 receiver 要怎麼設定。

**影響。** `FlinkInteractionFeatureJobMissing`、`ServiceErrorBudgetBurn` 等 critical alert 觸發後，沒有任何人會收到。25 的 crash loop 在預設部署下也不會有人知道。

**建議與驗收。** receiver 以 `*_FILE` 方式注入 webhook 或 PagerDuty，並加上 inhibit rules；每條 alert 都補上指向 `docs/operations` 的 `runbook_url`；chart 增加可選的 PrometheusRule。以 `amtool check-config` 驗證，並寫一個測試要求每條 alert 都有 runbook。規模 S；信心 高。

### 70．P1：Compose 預設提供的是對外開放、且不需驗證的 API

**證據。** Caddy 綁定 `"80:80"` 與 `"443:443"`（`docker-compose.yml:358-360`），也就是所有網路介面；其他服務都只綁 127.0.0.1。`.env.example` 預設 `API_API_KEY=` 為空，而 `gateway/api.py:130-131` 在 `auth_mode == "api_key"` 且沒有設定 key 時，直接 `authenticated = True`。Compose 預設 `ENVIRONMENT=development`，所以 production 的 fail-fast 不會觸發。此外 Redis 沒有密碼，MinIO 使用 `minioadmin`。

**影響。** 在 VM 上照 AGENTS.md 執行 `cp .env.example .env && ./startup.sh start` 之後，上傳、推薦、互動 API 會直接對公網開放且不需驗證；`/docs`、`/openapi.json`、`/readyz` 也同時公開。

**建議與驗收。** 沒有設定 key 時改為 fail-closed，只有明確設定 `SECURITY_AUTH_MODE=disabled` 才放行；startup.sh 預設要求提供 secret，或預設只綁 127.0.0.1；並提供一個能通過 production 驗證的 `deploy/compose/production.conf`（同時解決 23 的 preset 矛盾）。規模 S；信心 高（已讀原始碼確認）。

### 71．P2：供應鏈

- **未鎖定版本。** `requirements.runtime.txt:34-40` 的 7 個 OTEL 套件沒有指定版本，也沒有 hash 或 lockfile；`Dockerfile:60-64` 升級 pip、setuptools、cython 時也沒有鎖定版本。
- **使用 `:latest`。** Prometheus、Alertmanager、Grafana 都用 `:latest`（`docker-compose.yml:1345,1444,1458`），目前已經是 Prometheus 3.x，但 CI 的 promtool 是 v2.54.1。
- **EOL runtime。** Python 3.10 於 2026-10 EOL；Node 20（CI 與 `Code.Frontend/Dockerfile:1`）已於 2026-04 EOL；ESLint 8 也已 EOL。
- **已知 advisory（依版本號比對，未跑掃描工具）。**
  - 公開上傳路徑上：python-multipart 0.0.6（CVE-2024-24762）；fastapi 0.104.1 所帶的 starlette 0.27（CVE-2024-47874）。
  - torch 2.1.0（CVE-2025-32434），而且有 6 處 `torch.load` 沒有帶 `weights_only`（例如 `ranking.py:1435`、`two_tower.py:1815`）。
  - transformers 4.35.0、Pillow 10.0.1 也有已知 advisory。
- **流程面。** 沒有 Dependabot、pip-audit、npm audit、trivy 或 SBOM；GitHub Actions 只用 tag pin，沒有 SHA；workflow 沒有設定 `permissions:`。

**建議與驗收。** 用 pip-compile `--generate-hashes` 產生 lockfile；image 改用 digest；CI 加入 pip-audit、npm audit、trivy，並設定 fail 門檻；`torch.load` 一律加上 `weights_only=True`。升級 FastAPI 時，需同步規劃 pydantic v2 的遷移。規模 M；信心 高。

### 72．P2：Image 衛生

- production stage 繼承自 base，因此帶著 `build-essential`（`Dockerfile:21-23,100`）；從 PyPI 安裝的 torch 2.1.0 會連同 CUDA wheels 一起裝進 gateway 等只用 CPU 的服務。
- `COPY . /app/`（`:84`），但 `.dockerignore` 沒有排除 `.worktrees/`（約 21MB 的其他分支程式碼）與 `data/`（約 49MB 的 embeddings）。
- `gpu-runtime` stage（`:220`）使用 `nvidia/cuda:11.8-runtime-ubuntu20.04`，並在 focal 上以 apt 安裝 python3.10。研判這個 stage 無法 build，而且沒有任何地方使用。

**建議。** 分離 build 階段；只用 CPU 的服務改用 CPU 版 torch index；補齊 `.dockerignore`；刪除或修好 gpu stage。驗收時比較修改前後的 image 大小與 `docker history`。規模 S–M；信心 高（gpu stage 部分為中）。

### 73．P2：Helm 強化缺口

- 沒有 backend 的 NetworkPolicy（只有 Triton 有）；gateway 的 `/metrics` 免驗證，叢集內任何 pod 都能讀取。
- 只有全域的 `backend.affinity: {}`（`values.yaml:481`），各 workload 沒有 topologySpread 或 anti-affinity，PDB 擋不住單一節點故障。
- `readOnlyRootFilesystem: false`（`:66`），但實際需要寫入的目錄都已經是 emptyDir。
- 沒有 `values.schema.json`，secret 或 external host 留空也會靜默 render，要到 pod 執行時才 CrashLoop。
- image 只支援 `repo:tag`，預設的 `ci` 是可變的 tag，也不支援 digest（`_helpers.tpl:84-86`）。
- coordinator 只有單副本且沒有 PDB；Helm 的 topic 清單少了 `late-interaction-events`；worker 的 liveness 只打 metrics thread，event loop 卡住時不會被重啟。

**建議。** kubeconform 之外，再加上 kube-linter 或 polaris，以及自訂的 assert。規模 M；信心 高。

### 74．P2：Alert 品質與 dashboard 缺口

**證據。** 錯誤率的分母用 `clamp_min(rate, 1)`（`prometheus-rules.yml:4-8`），流量低於 1 rps 時錯誤率會被低估：在 0.04 rps 下，即使 100% 都是 5xx 也不會觸發。另外只有單一 window，沒有 multi-window burn rate。dashboard 只有 18 個 panel，沒有 ranking queue／runner、content pipeline、model release／rollout、fallback 比例等面板。

**建議。** 分母改為 `clamp_min(..., 1e-9)`，並加上最低流量條件；導入 1h/5m 與 6h/30m 的 burn rate；promtool 測試涵蓋所有 critical alert；補上述 dashboard。規模 S–M；信心 高。

### 75．P3：測試金字塔與 CI 效率

- **測試缺口。** gateway 的驗證與 rate limit middleware（`gateway/api.py:79-160`）沒有 unit test（`test_auth.py` 只有 2 個 bearer 測試）；`pit_state_exporter` 沒有測試；前端 Live 模式完全沒有測試，67 也因此漏掉。
- **不穩定的斷言與缺少設定。** `test_ranking_optimizations.py:3513,3533` 使用 `elapsed_ms < 25` 這類 wall-clock 斷言，在共用的 CI 機器上可能不穩定。專案也沒有 pytest 設定（markers、asyncio_mode、testpaths）。
- **CI 效率。** push 與 pull_request 會重複觸發，且沒有 `concurrency`；沒有 pip cache，4 個 job 各自下載 torch＋CUDA；backend 測試在 host 與 Docker 各跑一次；多數 job 沒有設定 timeout。
- **主機依賴。** 本次主機缺少 `sqlalchemy`、`onnx`、`faiss`，有 34 個模組無法收集（見第五節）。應提供一個能在主機上重現完整依賴的安裝目標。

規模 S。

### 76．P3：前端與 edge 細節

- `vite.config.js:30` 設定 `sourcemap: true`，source map 會對外提供。
- Caddyfile 沒有 CSP、HSTS、nosniff 等安全 header；`/docs` 與 `/openapi.json` 對外公開。
- 輪詢最長持續 6 分鐘，且沒有 AbortController；drawer 不支援 Escape 關閉，也沒有 focus 管理；`App.jsx` 單一元件樹內有 35 個 `useState`。
- CORS 有三套預設值，彼此不一致：`config.py:1681-1682` 的 `*`＋credentials 實際上沒有被 middleware 使用；`service_common.py:248-268` 預設 credentials=true；Compose 與 Helm 則是 false。

規模 S。

### 77．P3：文件與程式碼不一致

- `kubernetes-deployment.md` 的 topic 清單只有 5 個，Helm 實際有 13 個；文件把 feature-worker 列為部署服務，但它預設是關閉的；也沒有說明 Flink 這個外部依賴（見 01）。
- `slo-and-alerting.md:6` 用 feature-worker heartbeat 作為新鮮度 SLO（見 32）。
- README 第 12 行沒有提到 Flink；`startup.sh:33` 自稱是 production stack，實際卻執行 demo profile（見 23）。
- PROGRESS.md 的 10-08／10-09 entry 所描述的 retrieval 熱切換與 PIT parity gate，在預設路徑上實際沒有生效（見 58、41）。2026-10-20 的 entry 已註記這一點。

規模 S。

## 四、建議實作順序（修訂）

| 階段 | 工作包 | 完成條件 |
| --- | --- | --- |
| 0：立即修正（多數為 S） | 24、25、26、36、45、67、68、70，以及 04 的 payload_hash bug | 修正前產生的 retrieval evaluation 與 NDCG 門檻結果標記為無效，並重新產生 baseline<br>`startup.sh restart`、kill JobManager 之後，Flink generation 仍連續，且 conflict 為 0<br>超長欄位只進 DLQ<br>pit-production preset 能跑完一輪 ranking PIT 訓練<br>UI 的 Live 上傳成功<br>K8s 的 content-worker 能處理影片<br>未設 key 時不再免驗證 |
| A：讓量測可信 | 37、38、40、41、42、43、46、27、28，以及 09／10 的剩餘項 | gate 比較的對象是現役 release<br>PIT parity／quarantine 會 fail-closed<br>retrieval PIT 帶有真實的 seen 與 ranker_score<br>長尾 fixture 可以 PASS，A/A 偽失敗率低於 5%<br>在時鐘偏移與閒置使用者的 fixture 下，online/offline 一致 |
| B：預設部署可運作、可觀測 | 58、59、60、61、62、63、64、69、32、53、54，以及 01／02／05／06 的剩餘項與 22 | 遠端拓撲下 artifact 可以熱切換<br>單一依賴故障不會讓 edge 下線<br>counter 單調遞增<br>注入 DB 延遲後 p99 仍在預算內<br>critical alert 能送達且附有 runbook<br>處理途中 kill content-worker，job 最終仍會完成 |
| C：模型改善（需先完成 A） | 47、48、49、11、12、13、15、16、17、14、51、55、56 | 每一項都獨立消融，並固定 PIT manifest、catalog generation 與候選 budget<br>超過預先登記的效果門檻，且守住 p95 與成本，才升級 |
| D：容量與可維護性 | 29、30、39、52、07、19、20、18、23、31、33、34、44、50、65、66、71–77 | 百萬 key 下，每筆互動的 Redis 命令數為常數<br>PIT 的儲存與訓練時間有上界<br>完整鏈路在固定 Linux 主機上的負載測試達標<br>migration 可以獨立建出 schema<br>依賴有 lockfile 並經過掃描 |

不要一次啟用所有模型開關；每次只改一個主要因素，並記錄 dataset、catalog、model、feature、policy 與 experiment 的版本。階段 0 的 24 與 45 會改變過去 gate 結論的意義，修完後應重新產生 baseline，而不是沿用舊的數字。

## 五、本次驗證與限制

- **掃描方式。** 六個面向各自追蹤程式路徑，並核對首版 23 項。所有 P0，以及 04 的 hash bug 與以下 P1，都由我直接讀原始碼確認：36、38、41、45、47（程式路徑）、49（Dice 實作）、58、67、68、69、70。其餘項目以程式路徑追蹤為主，部分另有小型 probe。
- **小型 probe（在 scratchpad 執行，不改 repo）。**
  - 排序 gate：DIN Dice 合批前後的分數差異；gate 因單一小切片而判 INSUFFICIENT；全 view slate 的 NDCG 稀釋，以及隨機 challenger 模擬；gate bootstrap 耗時；ListNet 對非購買列 value head 的梯度。
  - 召回：legacy diversity filter 的截斷模擬。
  - 前端：axios 把 FormData 轉成 JSON 送出。
  - ingest：pydantic 補上 event_time 導致 hash 不同。
  - 跨語言：Jackson 2.15.3＋JDK17 與 Python canonical JSON 的差異。
  - gateway：主機 FastAPI 0.120 上，middleware 內的 HTTPException 會變成 500。
- **Compose。** `docker compose config -q` 通過（預設 profile 共 26 個服務，flink profile 共 30 個）。
- **主機 backend。** 以 Python 3.12.4 執行 `python -m pytest -q --ignore=tests/integration --continue-on-collection-errors`：394 passed、2 skipped、7 failed。7 個失敗全部是因為主機缺少 `onnx`（Phase C ONNX parity）或 `faiss`（visual index）；另有 34 個模組因缺少 `sqlalchemy` 而無法收集。沒有觀察到與程式邏輯相關的失敗，但這不是完整的 suite。
- **前端。** 在 Node 26.7.0 下，`npx vitest run` 14 項通過；`npm run lint` 通過（0 warning）。
- **未執行的驗證。**
  - Docker daemon 未啟動，因此沒有執行 Kafka／Flink／Compose smoke、image build、真實訓練、ONNX parity、影片處理或負載測試。
  - 主機上沒有 helm、promtool、kubeconform。
  - CVE 只依版本號比對已知 advisory，沒有跑掃描工具。
  - 各項影響的實際發生頻率取決於真實的資料與流量。本報告不宣稱任何效能或模型效果上的改善。
- **首版（2026-10-07）的驗證紀錄。** Compose config 與 profile 渲染、15 個 PIT／契約測試、46 個同步 ranking／DIN 測試、兩個 source-method probe，以及前端 lint／build。當時 Docker daemon 同樣不可用。
