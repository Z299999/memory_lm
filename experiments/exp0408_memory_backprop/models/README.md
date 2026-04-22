# Models

模型文件体积较大，不纳入 git 管理。克隆仓库后需手动下载。

## Student Model

| 字段 | 值 |
|------|----|
| HuggingFace ID | `Qwen/Qwen2.5-0.5B-Instruct` |
| 本地路径 | `models/student/` |
| 大小 | ~1 GB |
| 用途 | Forward 推理 + Backward 训练 |

### 下载方法

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir="models/student")
```

或安装依赖后运行：

```bash
pip install -r requirements.txt
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/student')"
```

下载完成后 `models/student/` 应包含：
- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json` / `merges.txt`
- `generation_config.json`
