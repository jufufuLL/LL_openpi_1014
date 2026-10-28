# LLbot 真机部署客户端

把训练好的 `pi05_llbot_lora` 模型部署到工控机（双臂 + 头部 + 双夹爪），支持**本机本地推理**与**远程 WebSocket 推理**两种方式。

文件：[llbot_offline_deploy.py](./llbot_offline_deploy.py)

---

## 1. 与训练侧严格对齐

客户端完全按 `src/openpi/policies/llbot_policy.py` 与 `convert_bag_to_lerobot.py` 实现，接口定义如下：

### observation.state / action：20 维

| 维度 | 索引 | 来源 / 说明 |
|------|------|------|
| 左臂关节 | 0–7 | `/joint_states` 的 `joint_la0`~`joint_la7` |
| 右臂关节 | 8–15 | `/joint_states` 的 `joint_ra0`~`joint_ra7` |
| 左夹爪 | 16 | `/left_gripper/joint_states` 的 `joint_lf`，归一化 0~1（0=张开，1=闭合） |
| 右夹爪 | 17 | `/right_gripper/joint_states` 的 `joint_rf`，归一化 0~1 |
| head_pan | 18 | **无 state 话题 → 恒填 0**（与转换脚本一致） |
| head_tilt | 19 | **无 state 话题 → 恒填 0** |

- 动作 → 状态用**增量(delta)**：双臂 16 维 + 头部 2 维（`make_bool_mask(8,8,-1,-1,2)`）
- 左右夹爪用**绝对值**
- `/joint_states` 里的关节顺序是乱的，客户端按名字重排为 `la0..la7, ra0..ra7`（缺名会直接报错，避免静默出错）

### 图像：3 路（键名必须与 policy 一致）

| 键名 | 话题 | 说明 |
|------|------|------|
| `stereo_left_eye` | `/stereo/left_eye/image_raw` | 双目左目 → `base_0_rgb` |
| `stereo_right_eye` | `/stereo/right_eye/image_raw` | 双目右目 → `left_wrist_0_rgb` |
| `right_hand` | `/right_hand/right_hand/color/image_raw` | 右手相机 → `right_wrist_0_rgb` |

> **图像预处理**：按训练转换脚本，把图像**直接拉伸到 224×224 RGB**（不做 letterbox）。模型内部还会 `resize_with_pad`，输入已是 224 时为不变换，从而与训练分布一致。

### 命令话题（下发）

| 话题 | 类型 | 内容 |
|------|------|------|
| `/dual_arm_forward_position_controller/commands` | `std_msgs/Float64MultiArray` | 双臂 16 维目标位置（弧度） |
| `/head_forward_position_controller/commands` | `std_msgs/Float64MultiArray` | 头部 2 维（裁剪到 `[-1.57,-0.2]~[1.57,0.35]`） |
| `/left_gripper/gripper_command` | `unix_msgs/GripperCommand` | `mode='parallel'`, `name=['gripper_left']`, `position∈[0,1]` |
| `/right_gripper/gripper_command` | `unix_msgs/GripperCommand` | `mode='parallel'`, `name=['gripper_right']`, `position∈[0,1]` |

夹爪消息的 `mode / name / velocity(1.0) / effort(0.5)` 与采集时遥操作下发的一致（已核对采集 bag 原始消息）。

---

## 2. 环境准备

**ROS 环境**（本机 ROS_DOMAIN_ID=53）：

```bash
source /opt/ros/humble/setup.bash
source /home/unix_ai/work/controller/install/setup.bash
export ROS_DOMAIN_ID=53
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/unix_ai/config/fastdds_no_shm.xml
```

**工控机客户端环境**（系统 Python 3.10 + ROS；工控机为 aarch64，
勿在其上安装 openpi/jax/torch——客户端只依赖纯 Python 的 openpi_client）：

```bash
pip3 install --user "numpy>=1.22.4,<2" msgpack websockets pillow dm-tree
pip3 install --user /home/unix_ai/userdata/VLA/LL_openpi_1014/packages/openpi-client

# 验证：应打印 client env OK + Python 3.10.x
source /opt/ros/humble/setup.bash
source /home/unix_ai/work/controller/install/setup.bash
python3 -c "import rclpy, cv_bridge, openpi_client; from unix_msgs.msg import GripperCommand; print('client env OK')"
```

> 注意：运行客户端时**不要 activate 任何 venv**。rclpy/unix_msgs 按 Python 3.10 编译，
> 在 3.11 venv 下会 ImportError。若提示无 pip3：`sudo apt install python3-pip`。

**云端/服务器 openpi 环境**（x86_64 + NVIDIA GPU；仅在服务器上跑 `--serve` 本地推理才需要）：

```bash
# 把整个 LL_openpi_1014 仓库（含 uv.lock）同步到服务器后：
cd LL_openpi_1014
uv venv --python 3.11
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

---

## 3. 启动机器人（话题检查）

确保以下话题有数据（相机由操作者手动启动）：

```bash
ros2 topic hz /joint_states
ros2 topic hz /left_gripper/joint_states
ros2 topic hz /right_gripper/joint_states
ros2 topic hz /stereo/left_eye/image_raw
ros2 topic hz /stereo/right_eye/image_raw
ros2 topic hz /right_hand/right_hand/color/image_raw
```

---

## 4. 运行部署

> 强烈建议先 **dry-run**（不加 `--move`）确认观测/推理正常，再真机运动。

### 方式 A：WebSocket 远程推理（推荐，模型在云端）

```bash
cd /home/unix_ai/userdata/VLA/LL_openpi_1014
# 用系统 python3 跑客户端（不要 activate .venv）
source /opt/ros/humble/setup.bash
source /home/unix_ai/work/controller/install/setup.bash

# 只推理不运动
python3 llbot_client/llbot_offline_deploy.py \
  --prompt "把红色方块放进篮子里" \
  --ws-host <服务器IP> --ws-port 8000 \
  --frames 50 --control-hz 10

# 真机运动
python3 llbot_client/llbot_offline_deploy.py \
  --prompt "把红色方块放进篮子里" \
  --ws-host <服务器IP> --ws-port 8000 \
  --replan-steps 10 --control-hz 10 --move
```

### 方式 B：本地推理（常驻服务模式；在 x86 GPU 服务器上运行）

> 工控机（aarch64）不支持本地推理，此方式仅在完成「云端/服务器 openpi 环境」的 GPU 服务器上使用。
> 终端 1 在服务器上跑（无需 ROS）；终端 2 在工控机上跑（系统 python + ROS，`--ws-host` 填服务器 IP）。

模型只需加载一次，之后客户端秒级连接：

```bash
# 终端1：常驻模型服务器（加载可能数分钟，之后常驻）
python3 llbot_client/llbot_offline_deploy.py \
  --checkpoint-dir checkpoints/pi05_llbot_lora \
  --config-name pi05_llbot_lora --serve

# 终端2：连接本地服务器推理
python3 llbot_client/llbot_offline_deploy.py \
  --prompt "把红色方块放进篮子里" \
  --ws-host localhost --ws-port 8001 --replan-steps 10 --move
```

### 方式 C：本地推理（直连，每次启动等加载，备用；同样仅在 x86 GPU 服务器上可用）

```bash
python3 llbot_client/llbot_offline_deploy.py \
  --prompt "把红色方块放进篮子里" \
  --checkpoint-dir checkpoints/pi05_llbot_lora \
  --config-name pi05_llbot_lora \
  --frames 200 --control-hz 10 --move
```

---

## 5. 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--prompt` | 必填 | 语言指令（须与训练时任务描述一致；`--serve` 模式可省略） |
| `--ws-host` | 无 | WebSocket 推理服务器地址（指定则使用远程推理） |
| `--ws-port` | 8000 | WebSocket 端口 |
| `--checkpoint-dir` | `checkpoints/pi05_llbot_lora` | 本地模型目录 |
| `--config-name` | `pi05_llbot_lora` | 训练配置名 |
| `--device` | `cuda` | 推理设备 |
| `--frames` | 0 | 控制步数，0=无限循环（Ctrl+C 停止） |
| `--control-hz` | 10 | 下发频率 |
| `--replan-steps` | 1 | 每次推理取前 N 步动作再重推（异步+平滑建议 5~10） |
| `--move` | 关 | 真正下发指令；不加则只推理（dry-run） |
| `--sync` | 关 | 同步推理：队列耗尽才推理，推理期间停顿（默认异步） |
| `--wait-timeout` | 30 | 等待相机/关节超时秒 |
| `--serve` | 关 | 常驻模式：加载模型后开启 WebSocket 服务（默认端口 8001） |

防回抽平滑（默认启用）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--no-smooth` | 关 | 禁用平滑（用于对比测试） |
| `--max-vel` | 0.12 | 单关节每步最大变化 rad/step（跳变兜底） |
| `--blend-steps` | 5 | 新块接入的混合窗口步数，0=关闭混合 |
| `--back-damp` | 0.15 | 回抽方向保留比例，0=完全不许回抽，1=不做方向阻尼 |

---

## 6. 防回抽平滑

异步推理时，新动作块基于「触发时刻」的观测预测，推理不准时接入点可能落在上一次已下发指令的**后面**，直接接上去就是动作瞬间往回跳（肉眼看到的**回抽**）。平滑器做两层处理：

1. **混合过渡**（`splice_with_blend`）：新块接入时不硬切，用升余弦权重在 `blend-steps` 步内把「旧计划的延续」与「新块」加权过渡。
2. **速度门控**（`gate`）：把期望增量投影到近期运动方向上，只压负向（回抽）分量保留 `back-damp`，垂直分量（横向纠偏）完整保留；另有逐关节 `|Δ| ≤ max-vel` 作为跳变兜底。

平滑只作用于**双臂 16 维**；**夹爪/头部直通**（夹爪要果断开合，头部行程小）。

运行日志标记：`[SMOOTH-BLEND]`（混合）、`[SMOOTH-GATE]`（限速）、`[ASYNC] ... splice_lag=...`（接入点滞后量）。结束打印 `[SUMMARY] 平滑器统计`。

调参建议：回抽仍明显就降 `--back-damp`（0.1/0.05）；接入点跳变明显就加 `--blend-steps`（7/10）；运动迟钝则提高 `--back-damp`（0.25）。`--replan-steps 1` 时混合窗口会退化，异步+平滑建议 `--replan-steps 5~10`。

---

## 7. Checkpoint 位置

- **常见情况：在云端**——用 WebSocket 远程推理（方式 A），质检机上无需保存权重。
- **若在本地**：放到 `/home/unix_ai/userdata/VLA/LL_openpi_1014/checkpoints` 下（默认 `--checkpoint-dir checkpoints/pi05_llbot_lora`）。

---

## 8. 故障排查

| 现象 | 处理 |
|------|------|
| `等待传感器超时` | 看日志末尾「缺少」列表：相机未启动→手动启动相机；关节/夹爪缺失→确认机器人 bringup 已起、`ROS_DOMAIN_ID=53` 一致 |
| `KeyError: /joint_states 缺少手臂关节` | 关节命名不是 `joint_la0-7/joint_ra0-7`，请核对机器人实际命名 |
| 夹爪不动作 | 确认 `/left_gripper/gripper_command` 类型为 `unix_msgs/GripperCommand`、`mode='parallel'`、`position∈[0,1]` |
| 推理后不动 | 未加 `--move`（dry-run）；或 `--prompt` 与训练任务描述不符 |
| 摄像头全黑/分布外 | 确认三路相机都有效（训练使用了右手相机），光照与采集时一致 |

### 快速核对命令

```bash
ros2 topic list | grep -E "joint_states|gripper|stereo|right_hand"
ros2 topic info -v /left_gripper/gripper_command
ros2 topic echo /joint_states --once --field name
ros2 topic echo /left_gripper/joint_states --once --field position
```
