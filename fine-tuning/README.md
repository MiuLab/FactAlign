## Setup

### Environment Setup

```bash
$ uenv cuda-12.1.0

$ uv sync

$ MAX_JOBS=20 uv pip install flash-attn --no-build-isolation --no-cache-dir
```