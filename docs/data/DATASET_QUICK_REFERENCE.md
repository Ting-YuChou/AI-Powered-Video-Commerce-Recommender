# Dataset 快速参考

## 🎯 哪些文件可以作为系统输入？

**所有 4 个 CSV 文件都可以作为系统输入！**

## 📦 文件清单

### ✅ 必需文件（系统核心功能）

1. **`products.csv`** (14MB, ~150,000 条)
   - 🔴 **必需** - 没有此文件系统无法运行
   - 包含产品目录信息
   - 推荐的目标产品

2. **`video_commerce_interactions_500k.csv`** (136MB, ~500,000 条)
   - 🔴 **必需** - 协同过滤的核心训练数据
   - 包含用户与产品的所有交互记录
   - 用于训练推荐模型

### ⭐ 重要文件（提升推荐质量）

3. **`users.csv`** (5.0MB, ~100,000 条)
   - 🟡 **重要** - 个性化推荐的基础
   - 包含用户基础信息和特征
   - 没有此文件仍可运行，但推荐质量下降

4. **`multimodal_features.csv`** (80MB, ~200,000 条) + **`video_embeddings_128d.npy`**
   - 🟢 **推荐** - 基于内容的推荐
   - `multimodal_features.csv`: 包含视频的元数据（音频、OCR、场景标签等）
   - `video_embeddings_128d.npy`: 包含视频的视觉嵌入向量（200,000 个视频，每个 128 维）
   - 没有这些文件仍可运行，但无法做基于内容的推荐

---

## 🚀 快速使用

### 最小配置（基本功能）
```bash
# 只加载必需文件
python scripts/load_dataset.py \
  --limit-products 10000 \
  --limit-interactions 50000
```

### 完整配置（最佳效果）
```bash
# 加载所有文件
python scripts/load_dataset.py
```

### 测试配置（快速验证）
```bash
# 限制数量进行测试
python scripts/load_dataset.py \
  --limit-users 1000 \
  --limit-products 5000 \
  --limit-interactions 10000 \
  --limit-content 1000
```

---

## 📊 数据用途总结

| 文件 | 系统用途 | 推荐算法 |
|------|----------|----------|
| `products.csv` | 产品目录、推荐目标 | 所有算法 |
| `interactions_500k.csv` | 训练数据、用户行为 | 协同过滤 |
| `users.csv` | 用户特征、个性化 | 个性化推荐 |
| `multimodal_features.csv` + `video_embeddings_128d.npy` | 视频特征、内容匹配 | 基于内容的推荐 |

---

## ⚡ 一句话总结

- **`products.csv`** + **`interactions_500k.csv`** = 系统可以运行 ✅
- **+ `users.csv`** = 支持个性化推荐 ⭐
- **+ `multimodal_features.csv` + `video_embeddings_128d.npy`** = 完整推荐系统 🎯

**建议**: 加载所有 4 个文件以获得最佳推荐效果！
