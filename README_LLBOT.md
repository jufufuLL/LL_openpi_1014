# LLbot π0.5 微调说明

本文档记录在 openpi 基础上为 **LLbot 机器人**接入 π0.5 微调所做的全部改动、使用方法与注意事项。

参考对象：旧项目 `VLA_old/openpi_cloud` 的 Haibot 接入方式；官方说明见 [README.md](README.md) 的 *Fine-tuning* 章节。

---

## 1. 机器人 / 数据集规格

| 项目 | 值 |
|------|-----|
| 机器人名称 | LLbot（旧机器人 Haibot 的升级款） |
| 状态维度 | 20 |
| 动作维度 | 20 |
| 相机 | `stereo_left_eye`、`stereo_right_eye`、`right_hand` |
| 数据帧率 | 15 FPS |
| 数据集格式 | LeRobot v2.1 |

**状态 / 动作 20 维布局（两臂关节从 0 开始编号）：**

| 索引 | 含义 |
|------|------|
| 0–7 | 左臂 `left_arm_joint_0` ~ `left_arm_joint_7` |
| 8–15 | 右臂 `right_arm_joint_0` ~ `right_arm_joint_7` |
| 16 | 左夹爪 `left_gripper` |
| 17 | 右夹爪 `right_gripper` |
| 18 | 头部 `head_pan` |
| 19 | 头部 `head_tilt` |

**图像 → 模型输入槽位映射**（π0.5 只支持 3 路图像）：

| 数据集相机 | 模型槽位 | 说明 |
|-----------|---------|------|
| `stereo_left_eye` | `base_0_rgb` | 双目左眼，第三人称主视角 |
| `stereo_right_eye` | `left_wrist_0_rgb` | 双目右眼 |
| `right_hand` | `right_wrist_0_rgb` | 右手相机 |

**动作 delta 掩码**：`make_bool_mask(8, 8, -1, -1, 2)`

- 左臂 8 关节、右臂 8 关节、头部 2 自由度 → **增量（delta）**
- 左夹爪、右夹爪 → **绝对值**

---

## 2. 新增文件

### `src/openpi/policies/llbot_policy.py`

LLbot 的数据映射，训练与推理共用：

- `LLbotInputs`：把 LeRobot 数据（`images` / `state` / `actions` / `prompt`）转换成模型输入；
  - 校验相机集合只能是 `stereo_left_eye`、`stereo_right_eye`、`right_hand`；
  - 完成 3 路相机到 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` 的映射；
  - 缺失相机时用零图填充，并在 `image_mask` 中置 `False`；
  - 图像统一转为 `HWC uint8`。
- `LLbotOutputs`：推理时把模型输出截断为前 20 维动作。
- `make_llbot_example()`：随机输入示例，可用于快速自测。

---

## 3. 修改文件

### `src/openpi/training/config.py`

1. 新增导入：

   ```python
   import openpi.policies.llbot_policy as llbot_policy
   ```

2. 新增 `LeRobotLLbotDataConfig`（位于 `LeRobotAlohaDataConfig` 之后、`LeRobotLiberoDataConfig` 之前）：

   - `repack_transforms`：把 LeRobot 字段重命名为模型内部字段
     - `observation.images.stereo_left_eye` → `images.stereo_left_eye`
     - `observation.images.stereo_right_eye` → `images.stereo_right_eye`
     - `observation.images.right_hand` → `images.right_hand`
     - `observation.state` → `state`
     - `action` → `actions`
     - `prompt` → `prompt`（配合 `prompt_from_task=True`）
   - `action_sequence_keys = ("action",)`
   - `data_transforms`：`LLbotInputs` / `LLbotOutputs` + `DeltaActions` / `AbsoluteActions`
   - `delta_action_mask = make_bool_mask(8, 8, -1, -1, 2)`

3. 新增训练配置 `pi05_llbot_lora`：

   | 参数 | 值 |
   |------|-----|
   | 模型 | `Pi0Config(pi05=True, action_horizon=10)` |
   | LoRA | `paligemma_variant="gemma_2b_lora"`，`action_expert_variant="gemma_300m_lora"` |
   | 数据 | `LeRobotLLbotDataConfig(repo_id="test_1012", assets=AssetsConfig(asset_id="test_1012"))` |
   | 预训练权重 | `gs://openpi-assets/checkpoints/pi05_base/params` |
   | `num_train_steps` | 10 000 |
   | `batch_size` | 16 |
   | `log_interval` | 50 |
   | `save_interval` / `keep_period` | 2500 / 2500 |
   | `ema_decay` | `None`（LoRA 关闭 EMA） |
   | `freeze_filter` | 由同款 LoRA `Pi0Config.get_freeze_filter()` 生成 |

> `repo_id` 与 `asset_id` **必须一致**，否则训练时找不到归一化统计量。

### `src/openpi/training/data_loader.py`

官方实现只从 HuggingFace Hub 加载数据集。现在改为**优先读取本地** `openpi/datasets/<repo_id>`：

```python
local_path = pathlib.Path.cwd() / "datasets" / repo_id
root = str(local_path) if local_path.exists() else None
```

- 本地目录存在：直接离线加载，不发网络请求；
- 不存在：`root=None`，退回官方 HuggingFace Hub 逻辑。

> 该路径基于**当前工作目录**，因此训练/计算统计量时需在 `openpi/` 根目录执行。

### `src/openpi/shared/download.py`

基座权重与 assets 的默认下载位置从 `~/.cache/openpi` 改为**仓库内 `openpi/models/`**：

```python
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = str(_REPO_ROOT / "models")
```

- 目录布局与 GCS 桶保持一致：`models/openpi-assets/checkpoints/...`
- 已存在的权重会被直接复用（`maybe_download` 命中即返回），**不会重复下载**；
- 仍可用环境变量 `OPENPI_DATA_HOME` 覆盖默认路径。

### `.gitignore`

新增忽略项：

```gitignore
# 预训练权重缓存目录（openpi.shared.download 默认下载到此）
models/
datasets/
```

`datasets/` 中被忽略的是本地数据集（含软链接），避免误提交大文件。

### 数据转换脚本 `data_collect/convert_bag_to_lerobot.py`

`CODEBASE_VERSION` 由 `"v1.0"` 修正为 `"v2.1"`。

原因：openpi 锁定的 LeRobot 版本（`0cf8648`，要求 v2.x）会拒绝加载 v1.0 格式的数据集并抛出
`BackwardCompatibilityError`。旧项目的转换脚本本来写的就是 `v2.1`，其余元数据结构一致。
现有数据集 `test_1012/meta/info.json` 也已同步修正。

---

## 4. 目录结构

```
openpi/
├── models/                                  # 新增：预训练权重 / assets（已 gitignore）
│   └── openpi-assets/
│       └── checkpoints/
│           ├── pi05_base/params/            # π0.5 基座
│           └── pi0_fast_libero/
├── datasets/                                # 新增：本地数据集入口（已 gitignore）
│   └── test_1012 -> ../../data_collect/bags/converted/test_1012
├── assets/                                  # 归一化统计量
│   └── pi05_llbot_lora/
│       └── test_1012/norm_stats.json
└── src/openpi/
    ├── policies/llbot_policy.py             # 新增
    ├── shared/download.py                   # 修改：默认缓存目录
    └── training/
        ├── config.py                        # 修改：数据配置 + 训练配置
        └── data_loader.py                   # 修改：本地数据集加载
```

---

## 5. 使用流程

### 5.1 环境

```bash
cd /home/ll/Desktop/VLA/openpi
uv sync
```

> 国内网络提示：清华 PyPI 镜像对 uv 会返回 403，可使用阿里云镜像：
> `UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/" uv sync`

### 5.2 准备数据集

1. 把采集的 bag 转换成 LeRobot 格式（输出默认在 `data_collect/bags/converted/<集合名>`）：

   ```bash
   cd /home/ll/Desktop/VLA/data_collect
   python3 convert_bag_to_lerobot.py --collection <集合名>
   ```

2. 在 `openpi/datasets/` 下建立软链接（或直接复制）：

   ```bash
   ln -sfn /home/ll/Desktop/VLA/data_collect/bags/converted/<集合名> \
           /home/ll/Desktop/VLA/openpi/datasets/<集合名>
   ```

3. 修改 `config.py` 中 `pi05_llbot_lora` 的 `repo_id`，以及与之一致的 `AssetsConfig.asset_id`。

### 5.3 预下载基座模型（克隆到新机器后）

训练配置中的 `weight_loader` 指向 `gs://openpi-assets/checkpoints/pi05_base/params`。
`maybe_download()` 会**先查本地** `openpi/models/...`，命中即直接返回、不会重复下载，
但新机器上建议先把权重拉好（`params` 约 **12GB**）。

**方式一：服务器可访问 GCS（推荐）**

```bash
cd /path/to/openpi

# openpi 对 gs://openpi-assets 会优先调用 gsutil，未安装则退回 gcsfs（可能遇到鉴权问题）
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))"
```

也可以直接用 gsutil：

```bash
mkdir -p models/openpi-assets/checkpoints
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base models/openpi-assets/checkpoints/
```

**方式二：服务器无法访问 GCS（国内服务器常见）**

从已有权重的机器同步（整个 `pi05_base` 目录）：

```bash
rsync -avP /path/to/local/openpi/models/openpi-assets/checkpoints/pi05_base/ \
    <user>@<server>:/path/to/openpi/models/openpi-assets/checkpoints/pi05_base/
```

或打包后 `scp` 传输。

**验证（应直接打印本地路径，不触发下载）**

```bash
cd /path/to/openpi
uv run python -c "
from openpi.shared.download import get_cache_dir
from openpi.shared import download
print('cache dir:', get_cache_dir())
print('params   :', download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))
"
```

期望第一部分输出 `<openpi>/models`；`params/` 下应包含
`_CHECKPOINT_METADATA`、`manifest.ocdbt`、`commit_success.txt`、`array_metadatas`、`d/` 等。

> 若服务器克隆的是**官方仓库**（不含 `download.py` 的改动），默认路径仍是 `~/.cache/openpi`。
> 此时可设置 `OPENPI_DATA_HOME=<openpi>/models`，或把权重放到
> `~/.cache/openpi/openpi-assets/checkpoints/` 下。
>
> 基座模型只需 `params`；`assets` 不需要下载——微调用的是本地数据集算出的
> `assets/pi05_llbot_lora/<repo_id>/norm_stats.json`。

### 5.4 计算归一化统计量

```bash
cd /home/ll/Desktop/VLA/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_llbot_lora
```

输出：`assets/pi05_llbot_lora/<repo_id>/norm_stats.json`

### 5.5 开始训练

```bash
cd /home/ll/Desktop/VLA/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_llbot_lora \
    --exp-name=llbot_lora_ft --overwrite
```

无 wandb 时追加 `--no-wandb-enabled`；中断后可用 `--resume` 继续。

检查点保存在 `checkpoints/pi05_llbot_lora/llbot_lora_ft/`。

### 5.6 推理（可选）

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_llbot_lora \
    --policy.dir=checkpoints/pi05_llbot_lora/llbot_lora_ft/<step>
```

推理端输入的 `state` 需为 20 维，图像 key 使用 `stereo_left_eye` / `stereo_right_eye` / `right_hand`，
与训练保持一致。

---

## 6. 注意事项

1. **显存**：π0.5 约 3B 参数，LoRA 实际显存占用约 24GB。本机 RTX 4060 Laptop（8GB）不足以训练，
   请在具备大显存的机器上执行训练；本机只用于改配置、算统计量和流程验证。
2. **必须在 `openpi/` 根目录运行**：`data_loader.py` 的本地查找基于 `Path.cwd()`。
3. **数据集格式**：必须是 LeRobot v2.x。用本仓库的 `convert_bag_to_lerobot.py` 转换即可。
4. **`repo_id` 与 `asset_id` 一致**：否则报 `Normalization stats not found`。
5. **头部状态**：采集中没有头部 state 话题，转换脚本将 `state[18:20]` 置 0；
   `std=0` 的维度经 openpi 归一化（分母 `+1e-6`）后为常数，不会产生 NaN。
   后续推理端也需在 18/19 维喂 0 以与训练分布一致。
6. **换正式数据集**：更新 `repo_id` / `asset_id` 后，必须**重新运行** `compute_norm_stats.py`。
7. **`discrete_state_input`**：当前沿用旧 Haibot 与官方 `pi05_full_droid_finetune` 的默认值（True）。
   若训练异常，可尝试 `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`
   （官方 `pi05_libero` 的写法）。
8. **权重位置**：如需恢复官方默认缓存路径，设置 `OPENPI_DATA_HOME=~/.cache/openpi` 即可。
