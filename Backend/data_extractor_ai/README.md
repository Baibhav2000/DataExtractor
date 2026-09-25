# Project setup

Run the following command to install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For setup in CPU-only environment run this
```
uv sync --extra cpu
```

For setup in GPU environment run this
```
uv sync --extra gpu --index-strategy unsafe-best-match
```
