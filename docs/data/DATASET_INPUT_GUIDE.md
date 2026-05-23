# Dataset 输入文件说明

本文档详细说明 `Dataset` 文件夹中哪些文件可以作为系统的输入，以及它们的用途和重要性。

## 📋 文件概览

| 文件名 | 大小 | 记录数 | 必需性 | 用途 |
|--------|------|--------|--------|------|
| `users.csv` | 5.0MB | ~100,000 | ⭐⭐⭐ 重要 | 用户基础信息和特征 |
| `products.csv` | 14MB | ~150,000 | ⭐⭐⭐ 必需 | 产品目录（推荐目标） |
| `video_commerce_interactions_500k.csv` | 136MB | ~500,000 | ⭐⭐⭐ 必需 | 用户交互历史（训练数据） |
| `multimodal_features.csv` | 80MB | ~200,000 | ⭐⭐ 推荐 | 视频内容特征（内容推荐） |
| `video_embeddings_128d.npy` | - | ~200,000 | ⭐⭐ 推荐 | 视频视觉嵌入向量（128维） |

---

## 📁 详细说明

### 1. ✅ `users.csv` - 用户数据

**必需性**: ⭐⭐⭐ **重要**（系统可以运行，但推荐质量会下降）

**用途**:
- 存储用户基础信息
- 用于个性化推荐
- 用户特征提取和用户画像构建

**包含字段**:
```
user_id, signup_date, country, platform, preferred_language,
marketing_source, membership_level
```

**系统如何使用**:
- 转换为 `UserFeatures` 模型
- 存储在 Redis FeatureStore 中
- 用于：
  - 个性化推荐算法
  - 用户偏好分析
  - 协同过滤的用户特征

**如果没有此文件**:
- 系统仍可运行
- 但无法进行个性化推荐
- 只能使用通用推荐（热门产品）
- **排序模型使用默认特征**，推荐质量下降 15-30%
- **详细原因**: 参见 `WHY_USERS_CSV_MATTERS.md`

---

### 2. ✅ `products.csv` - 产品目录

**必需性**: ⭐⭐⭐ **必需**（系统无法运行）

**用途**:
- 产品目录数据
- 推荐的目标产品
- 产品特征和元数据

**包含字段**:
```
product_id, title, brand, product_category, price_value, currency,
product_rating, availability, country_restrictions
```

**系统如何使用**:
- 转换为 `ProductData` 模型
- 存储在 Redis FeatureStore 中
- 生成产品嵌入向量
- 添加到 FAISS 向量搜索引擎
- 用于：
  - 所有推荐算法的基础数据
  - 产品相似度搜索
  - 推荐结果展示

**如果没有此文件**:
- ❌ **系统无法运行**
- 没有产品可推荐
- 推荐引擎无法工作

---

### 3. ✅ `video_commerce_interactions_500k.csv` - 交互历史

**必需性**: ⭐⭐⭐ **必需**（核心训练数据）

**用途**:
- 用户与产品的交互历史
- 协同过滤算法的训练数据
- 用户行为分析

**包含字段**:
```
interaction_id, user_id, video_id, product_id, creator_id, session_id,
timestamp, device_type, platform, watch_time_seconds, video_duration,
watch_percentage, liked, shared, commented, added_to_cart, purchased,
purchase_amount, scroll_velocity, dwell_time, session_position,
video_category, product_category, product_price, product_rating,
creator_followers, video_views, video_likes, has_audio_transcript,
has_price_tag_ocr, visual_embedding_similarity, audio_transcript_sentiment,
recommendation_source, cvr_prediction, gmv_contribution, user_age,
user_gender, user_location, user_purchase_history_value
```

**系统如何使用**:
- 转换为交互记录
- 存储在 Redis FeatureStore 中
- 用于：
  - **协同过滤模型训练**（最重要）
  - 用户偏好计算
  - 产品热度统计
  - 用户行为模式分析
  - 更新用户特征（交互次数、偏好类别等）

**如果没有此文件**:
- ❌ **协同过滤无法工作**
- 无法训练推荐模型
- 只能使用基于内容的推荐
- 推荐质量大幅下降

---

### 4. ⭐ `multimodal_features.csv` + `video_embeddings_128d.npy` - 视频内容特征

**必需性**: ⭐⭐ **推荐**（增强推荐质量）

**文件说明**:
- `multimodal_features.csv`: 包含视频的元数据（音频、OCR、场景标签等）
- `video_embeddings_128d.npy`: 包含视频的视觉嵌入向量（200,000 个视频，每个 128 维）
  - **位置**: 项目根目录（与 `Dataset` 文件夹同级）
  - **格式**: NumPy 数组，shape (200000, 128)，dtype float16
  - **索引对应**: 第 i 行对应 `multimodal_features.csv` 中第 i 行的 `video_id`

**用途**:
- 视频的多模态特征
- 基于内容的推荐
- 视频-产品相似度匹配

**包含字段**:
```
video_id, visual_embedding_128d, audio_transcript, audio_sentiment,
ocr_text, scene_labels, quality_score
```

**重要**: 视觉嵌入向量实际存储在 `video_embeddings_128d.npy` 文件中，而不是 CSV 的 base64 字段。

**系统如何使用**:
- 转换为 `ContentFeatures` 模型
- **从 `video_embeddings_128d.npy` 文件加载视觉嵌入向量**（128维 → 512维）
- 存储在 Redis FeatureStore 中
- 用于：
  - **基于内容的推荐**（Content-Based Filtering）
  - 视频-产品相似度搜索
  - 多模态特征匹配
  - 当用户观看视频时，推荐相似产品

**如果没有此文件**:
- ✅ 系统仍可运行
- 但无法进行基于内容的推荐
- 只能使用协同过滤和热门推荐
- 推荐多样性降低

---

## 🔄 数据流和依赖关系

```
┌─────────────────┐
│  users.csv      │ ──┐
└─────────────────┘   │
                      │
┌─────────────────┐   │    ┌──────────────────────┐
│ products.csv    │ ──┼───▶│  FeatureStore        │
└─────────────────┘   │    │  (Redis)             │
                      │    └──────────────────────┘
┌─────────────────┐   │              │
│ interactions    │ ──┘              │
│ _500k.csv       │                  │
└─────────────────┘                  │
                                     ▼
┌─────────────────┐         ┌──────────────────────┐
│ multimodal_     │ ───────▶│  VectorSearchEngine  │
│ features.csv    │         │  (FAISS)             │
└─────────────────┘         └──────────────────────┘
                                     │
                                     ▼
                            ┌──────────────────────┐
                            │  Recommendation     │
                            │  Engine             │
                            └──────────────────────┘
```

---

## 📊 最小运行配置

### 方案 1: 最小配置（基本功能）
```bash
✅ products.csv          # 必需
✅ video_commerce_interactions_500k.csv  # 必需（至少部分数据）
❌ users.csv            # 可选
❌ multimodal_features.csv  # 可选
```
**功能**: 基础推荐，无个性化

### 方案 2: 推荐配置（完整功能）
```bash
✅ products.csv
✅ video_commerce_interactions_500k.csv
✅ users.csv
✅ multimodal_features.csv
```
**功能**: 完整推荐系统，包含所有推荐算法

---

## 🎯 使用场景

### 场景 1: 冷启动（新用户）
**需要的文件**:
- ✅ `products.csv` - 显示产品
- ⭐ `multimodal_features.csv` - 基于内容推荐
- ❌ `users.csv` - 新用户无历史
- ❌ `interactions.csv` - 无交互历史

**推荐策略**: 基于内容的推荐 + 热门产品

### 场景 2: 有历史用户
**需要的文件**:
- ✅ `products.csv`
- ✅ `users.csv`
- ✅ `video_commerce_interactions_500k.csv`
- ⭐ `multimodal_features.csv`

**推荐策略**: 协同过滤 + 基于内容 + 个性化

### 场景 3: 视频推荐场景
**需要的文件**:
- ✅ `products.csv`
- ✅ `multimodal_features.csv` - **必需**
- ✅ `video_commerce_interactions_500k.csv`
- ✅ `users.csv`

**推荐策略**: 当用户观看视频时，使用视频特征匹配相似产品

---

## 💡 最佳实践

1. **首次测试**: 使用限制数量加载所有文件
   ```bash
   python scripts/load_dataset.py \
     --limit-users 1000 \
     --limit-products 5000 \
     --limit-interactions 10000 \
     --limit-content 1000
   ```

2. **生产环境**: 加载完整数据集
   ```bash
   python scripts/load_dataset.py
   ```

3. **快速原型**: 只加载必需文件
   - `products.csv` + `interactions.csv`（部分数据）

---

## ⚠️ 注意事项

1. **数据一致性**:
   - 确保 `interactions.csv` 中的 `user_id` 和 `product_id` 在对应的 CSV 中存在
   - 确保 `multimodal_features.csv` 中的 `video_id` 与交互数据中的 `video_id` 匹配
   - 确保 `video_embeddings_128d.npy` 的行数（200,000）与 `multimodal_features.csv` 的行数匹配
   - `.npy` 文件的第 i 行对应 CSV 文件的第 i 行（索引从 0 开始）

2. **内存使用**:
   - `interactions_500k.csv` (136MB) 加载后可能占用较多内存
   - 建议先用限制数量测试

3. **加载顺序**:
   - 系统会自动处理数据依赖关系
   - 但建议按顺序：products → users → interactions → features

4. **向量维度**:
   - `video_embeddings_128d.npy` 中的嵌入是 128 维（float16 格式）
   - 系统会自动转换为 512 维（float32）以兼容模型
   - `.npy` 文件的行索引对应 `multimodal_features.csv` 的行索引（第 i 行对应 video_id 的索引 i）

---

## 📝 总结

| 文件 | 必需性 | 主要用途 | 影响 |
|------|--------|----------|------|
| `products.csv` | 🔴 必需 | 产品目录 | 无此文件系统无法运行 |
| `interactions_500k.csv` | 🔴 必需 | 训练数据 | 无此文件协同过滤无法工作 |
| `users.csv` | 🟡 重要 | 个性化 | 无此文件推荐质量下降 |
| `multimodal_features.csv` + `video_embeddings_128d.npy` | 🟢 推荐 | 内容推荐 | 无此文件无法做基于内容的推荐 |

**建议**: 为了获得最佳推荐效果，应该加载所有四个文件。
