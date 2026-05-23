# 数据集使用指南 (Dataset Usage Guide)

本指南说明如何使用 `Dataset` 文件夹中的 CSV 数据集来测试视频商务推荐系统。

## 数据集文件说明

`Dataset` 文件夹包含以下 CSV 文件：

1. **users.csv** - 用户数据
   - 包含用户ID、注册日期、国家、平台、语言偏好等信息

2. **products.csv** - 产品数据
   - 包含产品ID、标题、品牌、类别、价格、评分等信息

3. **video_commerce_interactions_500k.csv** - 用户交互数据
   - 包含 50 万条用户与产品的交互记录
   - 包括观看时间、点赞、分享、购买等行为

4. **multimodal_features.csv** + **video_embeddings_128d.npy** - 视频多模态特征
   - `multimodal_features.csv`: 包含视频的元数据（音频转录、OCR文本、场景标签等）
   - `video_embeddings_128d.npy`: 包含视频的视觉嵌入向量（200,000 个视频，每个 128 维，float16 格式）
   - **位置**: `.npy` 文件在项目根目录（与 `Dataset` 文件夹同级）
   - **注意**: 视觉嵌入向量存储在 `.npy` 文件中，而不是 CSV 的 base64 字段
   - **索引对应**: `.npy` 文件的第 i 行对应 `multimodal_features.csv` 中第 i 行的 `video_id`

## 使用方法

### 方法 1: 使用独立脚本加载数据集（推荐）

在启动应用之前，先运行数据加载脚本：

```bash
# 加载所有数据
python scripts/load_dataset.py

# 限制加载数量（用于快速测试）
python scripts/load_dataset.py --limit-users 1000 --limit-products 5000 --limit-interactions 10000 --limit-content 1000

# 指定数据集目录
python scripts/load_dataset.py --dataset-dir Dataset

# 自定义 Redis 配置
python scripts/load_dataset.py --redis-host localhost --redis-port 6379
```

### 方法 2: 通过环境变量配置自动加载

设置环境变量，让应用在启动时自动加载 CSV 数据集：

```bash
# 启用 CSV 数据集加载
export DATA_USE_CSV_DATASET=true
export DATA_DATASET_DIR=Dataset

# 可选：限制加载数量
export DATA_CSV_LIMIT_USERS=1000
export DATA_CSV_LIMIT_PRODUCTS=5000
export DATA_CSV_LIMIT_INTERACTIONS=10000
export DATA_CSV_LIMIT_CONTENT=1000

# 启动应用
python app.py
```

### 方法 3: 在代码中直接调用

```python
import asyncio
from config import Config
from feature_store import FeatureStore
from vector_search import VectorSearchEngine
import data

async def load_data():
    config = Config()

    # 初始化组件
    feature_store = FeatureStore(config.redis_config)
    await feature_store.initialize()

    vector_search = VectorSearchEngine(config.vector_config)
    await vector_search.load_index()

    # 加载数据集
    summary = await data.load_dataset_from_csv(
        dataset_dir="Dataset",
        feature_store=feature_store,
        vector_search=vector_search,
        limit_users=1000,      # 可选限制
        limit_products=5000,   # 可选限制
        limit_interactions=10000,  # 可选限制
        limit_content=1000    # 可选限制
    )

    print(f"加载完成: {summary}")

asyncio.run(load_data())
```

## 数据加载过程

1. **读取 CSV 文件** - 使用 pandas 读取 CSV 文件
2. **数据转换** - 将 CSV 数据转换为系统所需的数据模型：
   - `users.csv` → `UserFeatures`
   - `products.csv` → `ProductData`
   - `video_commerce_interactions_500k.csv` → 交互记录
   - `multimodal_features.csv` → `ContentFeatures`
3. **嵌入向量解码** - 将 base64 编码的视觉嵌入向量解码为 numpy 数组
4. **用户特征更新** - 根据交互数据更新用户的统计信息（交互次数、偏好类别等）
5. **数据存储** - 将数据存储到：
   - FeatureStore (Redis) - 用户特征、交互记录、内容特征
   - VectorSearchEngine (FAISS) - 产品嵌入向量

## 注意事项

1. **内存使用**: 完整数据集（50万条交互）可能占用较多内存，建议先用限制数量测试
2. **Redis 要求**: 需要运行 Redis 服务器，数据会存储在 Redis 中
3. **向量维度**: 数据集中的视觉嵌入是 128 维，系统会自动填充到 512 维以兼容模型
4. **处理时间**: 完整数据集加载可能需要几分钟时间

## 验证数据加载

加载完成后，可以通过以下方式验证：

1. **检查 Redis 数据**:
```bash
redis-cli
> KEYS user:*
> KEYS product:*
> KEYS interaction:*
```

2. **通过 API 测试**:
```bash
# 获取推荐
curl -X POST "http://localhost:8000/api/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000000", "k": 10}'
```

3. **查看日志**: 数据加载过程会输出详细的日志信息

## 故障排除

### 问题：找不到 CSV 文件
- 确保 `Dataset` 文件夹在项目根目录
- 检查文件路径是否正确

### 问题：Redis 连接失败
- 确保 Redis 服务器正在运行: `redis-server`
- 检查 Redis 配置（host, port）

### 问题：内存不足
- 使用 `--limit-*` 参数限制加载数量
- 分批加载数据

### 问题：向量维度不匹配
- 系统会自动处理 128 维到 512 维的转换
- 如果仍有问题，检查 `vector_config.embedding_dim` 设置

## 性能优化建议

1. **首次加载**: 使用较小的限制数量进行测试
2. **生产环境**: 考虑分批加载或使用数据管道
3. **向量索引**: 加载完成后会自动保存 FAISS 索引，下次启动会更快

## 示例：快速测试

```bash
# 1. 启动 Redis（如果未运行）
redis-server

# 2. 加载少量数据进行测试
python scripts/load_dataset.py \
  --limit-users 100 \
  --limit-products 500 \
  --limit-interactions 1000 \
  --limit-content 100

# 3. 启动应用
python app.py

# 4. 测试 API
curl -X POST "http://localhost:8000/api/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000000", "k": 5}'
```
