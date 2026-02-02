# 水印嵌入提取功能模块


## 模块结构

```
func/
├── __init__.py           # 模块初始化
├── watermark_core.py     # 核心功能（模型加载、基础操作）
├── watermark_embedder.py # 水印嵌入器
├── watermark_extractor.py# 水印提取器
├── test_watermark.py     # 测试脚本
└── README.md             # 本文档
```

## 核心功能

### 1. 水印嵌入 (WatermarkEmbedder)

- **单图像处理**: 支持文件路径、PIL图像、numpy数组
- **批量处理**: 支持单个文件或嵌套文件夹
- **大图像处理**: 使用无重叠滑动窗口（200x200）遍历
- **小图像处理**: 填充到200x200中心区域，嵌入后裁剪回原大小

```python
from func import WatermarkEmbedder

# 初始化（随机生成水印密钥）
embedder = WatermarkEmbedder()
secret = embedder.get_secret_string()
print(f"水印密钥: {secret}")

# 单图像嵌入
watermarked = embedder.embed("input.png")
watermarked.save("output.png")

# 批量嵌入（支持嵌套文件夹）
result = embedder.embed_batch(
    input_path="./input_folder",
    output_path="./output_folder",
    recursive=True
)
```

### 2. 水印提取 (WatermarkExtractor)

- **多尺度提取**: 先resize到200x200，失败后使用滑动窗口
- **滑动窗口**: 支持 [200, 150, 100, 50] 四种窗口大小，50%重叠
- **投票机制**: 多个提取结果使用投票决定最终水印
- **早停机制**: 当前窗口大小提取成功则不继续更小窗口

```python
from func import WatermarkExtractor

# 初始化
extractor = WatermarkExtractor(
    expected_secret="101010...",  # 30位二进制
    similarity_threshold=0.9
)

# 提取水印
result = extractor.extract("watermarked.png", use_multiscale=True)
print(f"提取成功: {result['success']}")
print(f"提取密钥: {result['secret']}")
print(f"相似度: {result['similarity']:.2%}")

# 批量提取
result = extractor.extract_batch(
    input_path="./watermarked_folder",
    recursive=True
)
```

### 3. 测试脚本 (test_watermark.py)

测试各种正常和后处理情况下的水印有效性：

- **正常情况**: 无后处理
- **裁剪测试**: 不同比例和位置
- **旋转测试**: 不同角度
- **压缩测试**: 不同JPEG质量
- **缩放测试**: 不同缩放比例
- **模糊测试**: 不同模糊半径
- **噪声测试**: 不同噪声强度
- **拼接测试**: 水平/垂直拼接

```bash
# 运行测试
python func/test_watermark.py -i ./img/origin -o ./img/results  -b 8

# 参数说明
# -i, --input    测试图像路径或目录
# -o, --output   输出目录（默认: ./img/results）
# -s, --secret   30位二进制水印密钥（默认: 随机生成）
# -d, --device   运行设备（默认: cuda）
# -e,--eval_mode 进行后处理评估

python func/test_copy_move.py    -i ./img/origin512    -o ./img/result512_cm    -j result.json -b 64
#单独进行copy-move 评估
```


