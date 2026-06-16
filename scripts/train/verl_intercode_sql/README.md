# VERL InterCode SQL GRPO

This directory provides the VERL GRPO training path for InterCode SQL.

Files:
- `build_intercode_sql_grpo_dataset.py`: converts InterCode SQL JSON splits into VERL parquet files.
- `intercode_sql_agent_loop.py`: `AgentLoopBase` rollout loop backed by `InterCodeEnvironment`.
- `intercode_sql_agent_loop_config.yaml`: registers the SQL agent loop.
- `intercode_sql_reward.py`: reads environment reward from agent-loop extra fields.
- `run_qwen3_5_9b_intercode_sql_verl_grpo.sh`: launcher using `verl.trainer.main_ppo`.

Smoke:
```bash
DRY_RUN=1 bash scripts/train/verl_intercode_sql/run_qwen3_5_9b_intercode_sql_verl_grpo.sh
```

Use an already-running SQL service:
```bash
SQL_SERVICE_MODE=local \
SQL_HOST=127.0.0.1 SQL_PORT=3307 SQL_USER=admin SQL_PASSWORD=admin \
bash scripts/train/verl_intercode_sql/run_qwen3_5_9b_intercode_sql_verl_grpo.sh
```
