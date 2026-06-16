from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, List
from opd_evolver.benchmark.benchmark import Benchmark, LevelSpec
from opd_evolver.benchmark.common.env import Action, BasicInfo, Environment, Observation
DEFAULT_FAMILY_ENVS: dict[str, str] = {
    "room": "MiniHack-Room-5x5-v0",
    "maze": "MiniHack-MazeWalk-9x9-v0",
    "keyroom": "MiniHack-KeyRoom-Fixed-S5-v0",
    "river": "MiniHack-River-Narrow-v0",
}
DEFAULT_OBSERVATION_KEYS = (
    "tty_chars",
    "message",
    "blstats",
    "inv_strs",
    "inv_letters",
)
DEFAULT_MAX_STEPS = 80
PREMATURE_SUBMIT_REWARD = -0.2
MOVE_DIRECTIONS = {
    "north": "N",
    "south": "S",
    "east": "E",
    "west": "W",
    "northeast": "NE",
    "northwest": "NW",
    "southeast": "SE",
    "southwest": "SW",
    "n": "N",
    "s": "S",
    "e": "E",
    "w": "W",
    "ne": "NE",
    "nw": "NW",
    "se": "SE",
    "sw": "SW",
}
@dataclass(frozen=True)
class MiniHackTask:
    family: str
    env_id: str
    seed: int
    @property
    def level_id(self) -> str:
        safe_env_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.env_id)
        return f"minihack__{self.family}__{safe_env_id}__seed{self.seed}"
def _parse_seed_spec(spec: str | Iterable[int]) -> list[int]:
    if isinstance(spec, str):
        seeds: list[int] = []
        for part in spec.replace(",", " ").split():
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                if end < start:
                    raise ValueError(f"Invalid seed range: {part}")
                seeds.extend(range(start, end + 1))
            else:
                seeds.append(int(part))
        return sorted(set(seeds))
    return sorted(set(int(seed) for seed in spec))
def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)
def _decode_bytes_or_array(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        flat: list[int] = []
        def _collect(items: Any) -> None:
            if isinstance(items, list):
                for item in items:
                    _collect(item)
            elif isinstance(items, int):
                flat.append(items)
        _collect(value)
        if flat:
            return bytes(x for x in flat if 0 <= x <= 255).decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)
def _ascii_map(tty_chars: Any) -> str:
    if tty_chars is None:
        return ""
    if hasattr(tty_chars, "tolist"):
        tty_chars = tty_chars.tolist()
    lines: list[str] = []
    for row in tty_chars:
        chars: list[str] = []
        for cell in row:
            try:
                value = int(cell)
            except Exception:
                value = ord(str(cell)[0]) if str(cell) else 32
            chars.append(chr(value) if 0 <= value <= 255 else "?")
        lines.append("".join(chars).rstrip())
    return "\n".join(lines).rstrip()
def _inventory(inv_strs: Any) -> list[str]:
    if inv_strs is None:
        return []
    if hasattr(inv_strs, "tolist"):
        inv_strs = inv_strs.tolist()
    items = inv_strs if isinstance(inv_strs, list) else [inv_strs]
    decoded = [_decode_bytes_or_array(item).strip() for item in items]
    return [item for item in decoded if item]
def _inventory_letters(inv_letters: Any) -> list[str]:
    if inv_letters is None:
        return []
    if hasattr(inv_letters, "tolist"):
        inv_letters = inv_letters.tolist()
    flat: list[int] = []
    def _collect(items: Any) -> None:
        if isinstance(items, list):
            for item in items:
                _collect(item)
        elif isinstance(items, int):
            flat.append(items)
    _collect(inv_letters)
    return [chr(v) for v in flat if (ord("a") <= v <= ord("z")) or (ord("A") <= v <= ord("Z"))]
class MiniHackEnvironment(Environment):
    def __init__(
        self,
        level: LevelSpec,
        max_steps: int = DEFAULT_MAX_STEPS,
        observation_keys: tuple[str, ...] = DEFAULT_OBSERVATION_KEYS,
    ):
        self.level = level
        self.family = str(level["family"])
        self.env_id = str(level["env_id"])
        self.seed = int(level["seed"])
        self.max_steps = int(max_steps)
        self.observation_keys = tuple(observation_keys)
        self.env: Any = None
        self.steps = 0
        self.done = False
        self.cumulative_reward = 0.0
        self._action_index: dict[Any, int] = {}
        self._nethack: Any = None
        self._current_inv_strs: list[str] = []
        self._current_inv_letters: list[str] = []
    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=str(self.level["id"]),
            instruction=(
                f"Solve MiniHack environment {self.env_id} with seed {self.seed}. "
                "Navigate the map, collect/use required items, handle doors or rivers, "
                "and reach the goal."
            ),
            action_space=_compact_action_space(),
            max_steps=self.max_steps,
            meta_data={
                "benchmark": "minihack",
                "family": self.family,
                "env_id": self.env_id,
                "seed": self.seed,
                "tags": ["minihack", self.family],
            },
        )
    async def reset(self, seed: int | None = None) -> Observation:
        gym, nethack = self._load_dependencies()
        self._nethack = nethack
        self.env = self._make_env(gym)
        self._build_action_index()
        self.steps = 0
        self.done = False
        self.cumulative_reward = 0.0
        reset_seed = self.seed if seed is None else int(seed)
        try:
            reset_result = self.env.reset(seed=reset_seed)
        except TypeError:
            if hasattr(self.env, "seed"):
                self.env.seed(reset_seed)
            reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs, info = reset_result, {}
        return self._format_observation(obs, info, reward=0.0, terminated=False, truncated=False)
    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            return {"error": "Environment already finished"}, 0.0, True, {"error": "already_done"}
        if self.env is None:
            return {"error": "Environment has not been reset"}, 0.0, True, {"error": "not_reset"}
        action_name = str(action.get("action", "")).strip().lower()
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        if action_name == "submit":
            success = self.cumulative_reward > 0.0
            self.done = True
            if not success:
                return (
                    {
                        "error": "premature_submit",
                        "message": "Submit is only valid after reaching the goal or receiving positive reward.",
                        "success": False,
                        "current_step": self.steps,
                        "max_steps": self.max_steps,
                        "cumulative_reward": self.cumulative_reward,
                    },
                    PREMATURE_SUBMIT_REWARD,
                    True,
                    {"submitted": True, "success": False, "error": "premature_submit"},
                )
            return (
                {
                    "message": "MiniHack task submitted.",
                    "success": success,
                    "current_step": self.steps,
                    "max_steps": self.max_steps,
                    "cumulative_reward": self.cumulative_reward,
                },
                1.0,
                True,
                {"submitted": True, "success": success},
            )
        try:
            indices = self._resolve_action_indices(action_name, params)
        except ValueError as exc:
            obs = {
                "error": str(exc),
                "current_step": self.steps,
                "max_steps": self.max_steps,
                "cumulative_reward": self.cumulative_reward,
            }
            return obs, 0.0, False, {"error": str(exc)}
        total_reward = 0.0
        obs: Any = {}
        info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for index in indices:
            obs, reward, terminated, truncated, info = self._gym_step(index)
            reward_f = float(reward)
            total_reward += reward_f
            self.cumulative_reward += reward_f
            self.steps += 1
            if terminated or truncated or self.steps >= self.max_steps:
                break
        self.done = bool(terminated or truncated or self.steps >= self.max_steps)
        formatted = self._format_observation(obs, info, total_reward, terminated, truncated)
        step_info = {
            "reward": total_reward,
            "cumulative_reward": self.cumulative_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(terminated or self.cumulative_reward > 0.0),
        }
        return formatted, total_reward, self.done, step_info
    async def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
    def _load_dependencies(self) -> tuple[Any, Any]:
        try:
            import gymnasium as gym
            import minihack
            from nle import nethack
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MiniHack dependencies are required on the runtime server. "
                "Install gymnasium, minihack, and nle in the same environment used for eval."
            ) from exc
        self._patch_minihack_spaces(gym)
        return gym, nethack
    def _patch_minihack_spaces(self, gym: Any) -> None:
        try:
            import minihack.base as minihack_base
        except Exception:
            return
        patched = []
        changed = False
        for key, space in list(minihack_base.NLE_SPACE_ITEMS):
            if isinstance(space, gym.spaces.Space):
                patched.append((key, space))
                continue
            if hasattr(space, "low") and hasattr(space, "high"):
                patched.append(
                    (
                        key,
                        gym.spaces.Box(
                            low=space.low,
                            high=space.high,
                            shape=getattr(space, "shape", None),
                            dtype=getattr(space, "dtype", None),
                        ),
                    )
                )
                changed = True
                continue
            patched.append((key, space))
        if changed:
            minihack_base.NLE_SPACE_ITEMS = tuple(patched)
    def _make_env(self, gym: Any) -> Any:
        actions = self._default_actions()
        kwargs = {
            "observation_keys": self.observation_keys,
            "actions": actions,
            "savedir": None,
        }
        try:
            return self._make_registered_env(gym, kwargs)
        except TypeError:
            kwargs.pop("savedir", None)
            try:
                return self._make_registered_env(gym, kwargs)
            except TypeError:
                kwargs.pop("actions", None)
                return self._make_registered_env(gym, kwargs)
        except KeyError:
            pass
        try:
            return gym.make(
                self.env_id,
                observation_keys=self.observation_keys,
                actions=actions,
                savedir=None,
            )
        except TypeError:
            try:
                return gym.make(
                    self.env_id,
                    observation_keys=self.observation_keys,
                    actions=actions,
                )
            except TypeError:
                return gym.make(self.env_id, observation_keys=self.observation_keys)
        except Exception:
            return gym.make(self.env_id)
    def _make_registered_env(self, gym: Any, kwargs: dict[str, Any]) -> Any:
        spec = gym.envs.registry[self.env_id]
        entry_point = spec.entry_point
        env_kwargs = dict(getattr(spec, "kwargs", {}) or {})
        env_kwargs.update(kwargs)
        if callable(entry_point):
            creator = entry_point
        else:
            from gymnasium.envs.registration import load_env_creator
            creator = load_env_creator(str(entry_point))
        return creator(**env_kwargs)
    def _default_actions(self) -> list[Any]:
        actions: list[Any] = [
            self._nethack.CompassDirection.N,
            self._nethack.CompassDirection.S,
            self._nethack.CompassDirection.E,
            self._nethack.CompassDirection.W,
            self._nethack.CompassDirection.NE,
            self._nethack.CompassDirection.NW,
            self._nethack.CompassDirection.SE,
            self._nethack.CompassDirection.SW,
            self._nethack.Command.PICKUP,
            self._nethack.Command.APPLY,
            self._nethack.Command.OPEN,
            self._nethack.Command.SEARCH,
            self._nethack.MiscDirection.UP,
        ]
        try:
            actions.append(self._wait_command())
        except ValueError:
            pass
        covered: set[int] = set()
        for a in actions:
            v = getattr(a, "value", None)
            if v is None:
                try:
                    v = int(a)
                except (TypeError, ValueError):
                    continue
            covered.add(v)
        for char_ord in range(ord("a"), ord("z") + 1):
            if char_ord not in covered:
                actions.append(char_ord)
        return actions
    def _build_action_index(self) -> None:
        actions = list(getattr(getattr(self.env, "unwrapped", self.env), "actions", []))
        self._action_index = {action: idx for idx, action in enumerate(actions)}
    def _resolve_action_indices(self, action_name: str, params: dict[str, Any]) -> list[int]:
        if action_name == "move":
            direction = self._direction(params.get("direction"))
            return [self._index_for(self._nethack.CompassDirection[direction])]
        if action_name == "open":
            direction = self._direction(params.get("direction"))
            return [
                self._index_for(self._nethack.Command.OPEN),
                self._index_for(self._nethack.CompassDirection[direction]),
            ]
        if action_name == "pickup":
            return [self._index_for(self._nethack.Command.PICKUP)]
        if action_name == "apply":
            direction_raw = params.get("direction")
            key_letter = self._find_item_letter("key")
            if direction_raw and key_letter:
                dir_str = self._direction(direction_raw)
                key_idx = self._index_for_ascii(ord(key_letter))
                if key_idx is not None:
                    return [
                        self._index_for(self._nethack.Command.APPLY),
                        key_idx,
                        self._index_for(self._nethack.CompassDirection[dir_str]),
                    ]
            return [self._index_for(self._nethack.Command.APPLY)]
        if action_name == "climb_up":
            return [self._index_for(self._nethack.MiscDirection.UP)]
        if action_name == "search":
            return [self._index_for(self._nethack.Command.SEARCH)]
        if action_name == "wait":
            return [self._index_for(self._wait_command())]
        raise ValueError(f"Unknown MiniHack action: {action_name}")
    def _find_item_letter(self, *keywords: str) -> str | None:
        for letter, desc in zip(self._current_inv_letters, self._current_inv_strs):
            lower = desc.lower()
            if any(kw.lower() in lower for kw in keywords):
                return letter
        return None
    def _index_for_ascii(self, char_value: int) -> int | None:
        for action, idx in self._action_index.items():
            action_val = getattr(action, "value", None)
            if action_val is None:
                try:
                    action_val = int(action)
                except (TypeError, ValueError):
                    continue
            if action_val == char_value:
                return idx
        return None
    def _direction(self, raw: Any) -> str:
        key = str(raw or "").strip().lower()
        if key not in MOVE_DIRECTIONS:
            raise ValueError(f"Invalid direction: {raw}")
        return MOVE_DIRECTIONS[key]
    def _wait_command(self) -> Any:
        if hasattr(self._nethack, "MiscDirection") and hasattr(self._nethack.MiscDirection, "WAIT"):
            return self._nethack.MiscDirection.WAIT
        for name in ("WAIT", "NOOP"):
            if hasattr(self._nethack.Command, name):
                return getattr(self._nethack.Command, name)
        raise ValueError("MiniHack action 'wait' is not available in this NLE build")
    def _index_for(self, nethack_action: Any) -> int:
        if nethack_action not in self._action_index:
            target_value = getattr(nethack_action, "value", None)
            target_name = getattr(nethack_action, "name", None)
            for action, idx in self._action_index.items():
                if target_value is not None and getattr(action, "value", None) == target_value:
                    return idx
                if target_name is not None and getattr(action, "name", None) == target_name:
                    return idx
            raise ValueError(f"MiniHack action is not enabled for {self.env_id}: {nethack_action}")
        return self._action_index[nethack_action]
    def _gym_step(self, index: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.env.step(index)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, bool(terminated), bool(truncated), dict(info or {})
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, bool(done), False, dict(info or {})
        raise RuntimeError(f"Unexpected Gymnasium step result: {type(result)}")
    def _format_observation(
        self,
        obs: Any,
        info: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Observation:
        obs_dict = obs if isinstance(obs, dict) else {}
        inv_strs = _inventory(obs_dict.get("inv_strs"))
        inv_letters = _inventory_letters(obs_dict.get("inv_letters"))
        self._current_inv_strs = inv_strs
        self._current_inv_letters = inv_letters
        return {
            "map": _ascii_map(obs_dict.get("tty_chars")),
            "message": _decode_bytes_or_array(obs_dict.get("message")),
            "inventory": inv_strs,
            "blstats": _json_safe(obs_dict.get("blstats")),
            "reward": float(reward),
            "cumulative_reward": float(self.cumulative_reward),
            "current_step": self.steps,
            "max_steps": self.max_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _json_safe(info or {}),
        }
def _compact_action_space() -> str:
    return (
        "MiniHack navigation actions. Reply with exactly one JSON object.\n"
        'Actions:\n'
        '  - {"action":"move","params":{"direction":"north|south|east|west|northeast|northwest|southeast|southwest"}}\n'
        '  - {"action":"pickup","params":{}}\n'
        '  - {"action":"apply","params":{"direction":"north|south|east|west|northeast|northwest|southeast|southwest"}}\n'
        '  - {"action":"open","params":{"direction":"north|south|east|west|northeast|northwest|southeast|southwest"}}\n'
        '  - {"action":"climb_up","params":{}}\n'
        '  - {"action":"search","params":{}}\n'
        '  - {"action":"wait","params":{}}\n'
        '  - {"action":"submit","params":{}}\n'
        "Use submit only after the observation clearly shows success or positive cumulative reward. "
        "Premature submit ends the episode with a penalty."
    )
class MiniHackBenchmark(Benchmark):
    def __init__(
        self,
        families: Iterable[str] | None = None,
        env_ids: Iterable[str] | None = None,
        seeds: str | Iterable[int] = "0",
        max_steps: int = DEFAULT_MAX_STEPS,
        observation_keys: tuple[str, ...] = DEFAULT_OBSERVATION_KEYS,
    ):
        family_source = DEFAULT_FAMILY_ENVS if families is None else families
        self.families = [str(f).strip().lower() for f in family_source if str(f).strip()]
        unknown = [family for family in self.families if family not in DEFAULT_FAMILY_ENVS]
        if unknown:
            raise ValueError(f"Unknown MiniHack families: {unknown}. Known: {sorted(DEFAULT_FAMILY_ENVS)}")
        self.extra_env_ids = [str(env_id).strip() for env_id in (env_ids or []) if str(env_id).strip()]
        self.seeds = _parse_seed_spec(seeds)
        self.max_steps = int(max_steps)
        self.observation_keys = tuple(observation_keys)
        self.tasks = self._build_tasks()
    def _build_tasks(self) -> list[MiniHackTask]:
        env_pairs: list[tuple[str, str]] = [
            (family, DEFAULT_FAMILY_ENVS[family]) for family in self.families
        ]
        env_pairs.extend(("custom", env_id) for env_id in self.extra_env_ids)
        tasks = [
            MiniHackTask(family=family, env_id=env_id, seed=seed)
            for family, env_id in env_pairs
            for seed in self.seeds
        ]
        if not tasks:
            raise ValueError("MiniHackBenchmark requires at least one env and one seed")
        return tasks
    def list_levels(self) -> List[LevelSpec]:
        return [
            {
                "id": task.level_id,
                "family": task.family,
                "env_id": task.env_id,
                "seed": task.seed,
                "index": idx,
            }
            for idx, task in enumerate(self.tasks)
        ]
    def make_env(self, level: LevelSpec) -> Environment:
        return MiniHackEnvironment(
            level=level,
            max_steps=self.max_steps,
            observation_keys=self.observation_keys,
        )
