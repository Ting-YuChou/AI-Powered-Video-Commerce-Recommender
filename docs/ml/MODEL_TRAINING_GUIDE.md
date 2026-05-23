# 模型训练和推理流程说明

## 📋 核心问题回答

### Q1: 模型训练完成后会保存下来做后续推理吗？

**答案：❌ 不会保存到磁盘，只在内存中**

- 协同过滤模型（CF）的参数保存在**内存**中
- 每次应用重启后需要**重新训练**
- 模型参数包括：
  - `user_features_matrix` - 用户特征矩阵
  - `item_features_matrix` - 产品特征矩阵
  - `user_mapping` - 用户ID映射
  - `item_mapping` - 产品ID映射

### Q2: 是一边跑一边训练，还是先训练完才能使用？

**答案：先训练完才能使用（启动时训练）**

- 应用启动时（`startup_event`）会先训练模型
- 训练完成后设置 `is_trained = True`
- 只有训练完成后才能进行推荐推理
- **不是**一边运行一边训练

### Q3: 需要先训练完推荐模型才能跑吗？

**答案：是的，但系统有容错机制**

- 如果模型未训练（`is_trained = False`）：
  - 协同过滤推荐返回**空列表**
  - 系统仍可使用其他推荐源（内容推荐、热门推荐）
  - 但推荐质量会下降

---

## 🔄 完整流程

### 1. 数据加载阶段

```python
# data.py - load_dataset_from_csv()
# 1. 从 CSV 加载交互数据
interactions = load_interactions_from_csv("video_commerce_interactions_500k.csv")

# 2. 存储到 Redis
for interaction in interactions:
    await feature_store.log_user_interaction(...)
    # 交互数据被存储到 Redis 的 "global_interactions" 列表
```

**存储位置**：
- Redis Key: `global_interactions`
- 类型: List（最多保留 10,000 条）
- TTL: 7 天

---

### 2. 应用启动阶段

```python
# app.py - startup_event()
@app.on_event("startup")
async def startup_event():
    # 1. 初始化组件
    feature_store = FeatureStore(...)
    recommendation_engine = RecommendationEngine(...)

    # 2. 加载模型（会触发训练）
    await recommendation_engine.load_models()
    # ↓
    # recommender.py - load_models()
    async def load_models(self):
        # 从 Redis 加载交互数据并训练
        await self._update_models_from_interactions()
```

---

### 3. 模型训练阶段

```python
# recommender.py - _update_models_from_interactions()
async def _update_models_from_interactions(self):
    # 1. 从 Redis 获取最近 10,000 条交互
    interactions_data = await self.feature_store.redis_client.lrange(
        "global_interactions", 0, 9999
    )

    # 2. 训练协同过滤模型
    if interactions:
        await self.cf_engine.train_model(interactions)
        # ↓
        # recommender.py - train_model()
        async def train_model(self, interactions):
            # 构建用户-产品交互矩阵
            await self._build_interaction_matrix(interactions)

            # 训练 NMF 模型
            self.user_features_matrix = self.model.fit_transform(self.user_item_matrix)
            self.item_features_matrix = self.model.components_.T

            # 标记为已训练
            self.is_trained = True
```

**训练时间**：
- 10,000 条交互：约 1-5 秒
- 50,000 条交互：约 5-15 秒
- 500,000 条交互：约 30-60 秒

---

### 4. 推理阶段（推荐请求）

```python
# app.py - get_recommendations()
@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    # 1. 生成候选产品
    candidates = await recommendation_engine.generate_candidates(...)
    # ↓
    # recommender.py - generate_candidates()
    async def generate_candidates(self, ...):
        # 检查模型是否已训练
        if self.cf_engine.is_trained:
            # 使用内存中的模型参数进行推理
            cf_candidates = await self.cf_engine.get_user_recommendations(...)
            # ↓
            # 使用 user_features_matrix 和 item_features_matrix
            # 计算用户-产品相似度分数
        else:
            # 模型未训练，返回空列表
            return []
```

---

## 📊 数据流图

```
┌─────────────────────────────────────┐
│  1. 数据加载（首次）                │
│  video_commerce_interactions_500k  │
│           .csv                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. 存储到 Redis                     │
│  Key: "global_interactions"         │
│  Type: List (最多 10,000 条)         │
│  TTL: 7 天                          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. 应用启动                         │
│  startup_event()                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. 模型训练（每次启动）              │
│  load_models()                      │
│  ↓                                  │
│  从 Redis 加载交互数据               │
│  ↓                                  │
│  训练 NMF 模型                      │
│  ↓                                  │
│  保存到内存：                        │
│  - user_features_matrix            │
│  - item_features_matrix             │
│  - is_trained = True                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. 推理（推荐请求）                 │
│  使用内存中的模型参数                │
│  计算推荐分数                        │
└─────────────────────────────────────┘
```

---

## ⚠️ 重要特点

### 1. 模型不持久化

```python
# ❌ 没有保存到磁盘的代码
# 模型参数只存在于内存中

# 每次重启应用：
# 1. 模型参数丢失
# 2. 需要重新从 Redis 加载数据
# 3. 重新训练模型
```

**影响**：
- ✅ 优点：模型总是使用最新数据
- ❌ 缺点：每次启动需要重新训练（耗时）

---

### 2. 训练时机

```python
# 只在以下情况训练：
# 1. 应用启动时（startup_event）
# 2. 手动调用 train_model()

# ❌ 不会自动重新训练
# - 新交互写入 Redis 后，模型不会自动更新
# - 需要重启应用才能使用新数据训练
```

**影响**：
- 模型可能使用**过时的数据**
- 需要定期重启应用以更新模型

---

### 3. 容错机制

```python
# recommender.py - get_user_recommendations()
if not self.is_trained or self.user_features_matrix is None:
    logger.warning("CF model not trained, returning empty recommendations")
    return []  # 或返回热门产品
```

**影响**：
- 即使模型未训练，系统仍可运行
- 但只能使用其他推荐源（内容推荐、热门推荐）

---

## 🔧 优化建议

### 方案 1: 添加模型持久化（推荐）

```python
# 在 recommender.py 中添加保存/加载功能

async def save_model(self, model_path: str):
    """保存模型到磁盘"""
    model_data = {
        'user_features_matrix': self.user_features_matrix,
        'item_features_matrix': self.item_features_matrix,
        'user_mapping': self.user_mapping,
        'item_mapping': self.item_mapping,
        'reverse_item_mapping': self.reverse_item_mapping,
        'last_training_time': self.last_training_time
    }
    np.savez_compressed(model_path, **model_data)
    logger.info(f"Model saved to {model_path}")

async def load_model(self, model_path: str):
    """从磁盘加载模型"""
    if Path(model_path).exists():
        model_data = np.load(model_path, allow_pickle=True)
        self.user_features_matrix = model_data['user_features_matrix']
        self.item_features_matrix = model_data['item_features_matrix']
        self.user_mapping = model_data['user_mapping'].item()
        self.item_mapping = model_data['item_mapping'].item()
        self.reverse_item_mapping = model_data['reverse_item_mapping'].item()
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")
```

### 方案 2: 增量训练

```python
# 定期（如每小时）重新训练模型
# 使用后台任务自动更新

async def periodic_model_update(self):
    """定期更新模型"""
    while True:
        await asyncio.sleep(3600)  # 每小时
        await self._update_models_from_interactions()
```

### 方案 3: 使用更高效的模型

```python
# 使用在线学习算法（如 SGD）
# 可以增量更新，不需要重新训练整个模型
```

---

## 📝 总结

| 问题 | 答案 |
|------|------|
| 模型会保存吗？ | ❌ 不保存到磁盘，只在内存中 |
| 训练时机？ | ✅ 应用启动时训练一次 |
| 需要先训练才能用吗？ | ⚠️ 建议先训练，但有容错机制 |
| 会一边跑一边训练吗？ | ❌ 不会，只在启动时训练 |
| 新交互会自动更新模型吗？ | ❌ 不会，需要重启应用 |

**当前实现**：
- ✅ 简单直接
- ✅ 总是使用最新数据（重启后）
- ❌ 每次启动需要重新训练
- ❌ 模型不持久化

**建议**：
- 添加模型持久化功能
- 实现增量训练或定期更新
- 考虑使用在线学习算法
