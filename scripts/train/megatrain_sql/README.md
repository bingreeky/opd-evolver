# MegaTrain SQL GRPO

This directory contains the remote-only GRPO preparation assets for InterCode SQL.

Files:
- `build_sql_grpo_dataset.py`: converts merged train/test JSON splits into VERL parquet files.
- `sql_intercode_interaction.py`: multi-turn VERL interaction backed by the existing InterCode SQL environment.
- `sql_interaction_config.yaml`: registers the SQL interaction with VERL.
- `run_qwen3_5_9b_sql_megatrain.sh`: launcher that ensures editable `verl`, starts MySQL, builds parquet files, and calls the reference MegaTrain script.

Usage:
```bash
cd .
bash scripts/train/megatrain_sql/run_qwen3_5_9b_sql_megatrain.sh
```

Dry run:
```bash
cd .
DRY_RUN=1 bash scripts/train/megatrain_sql/run_qwen3_5_9b_sql_megatrain.sh
```

Use an existing local MySQL-compatible InterCode SQL service instead of starting
the docker container:
```bash
SQL_SERVICE_MODE=local \
SQL_HOST=127.0.0.1 SQL_PORT=3307 SQL_USER=admin SQL_PASSWORD=admin \
bash scripts/train/megatrain_sql/run_qwen3_5_9b_sql_megatrain.sh
```
