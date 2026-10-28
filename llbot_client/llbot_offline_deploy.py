#!/usr/bin/env python3
"""LLbot 真机部署（工控机本地推理 或 WebSocket 远程推理）

与训练侧严格对齐（依据 openpi/policies/llbot_policy.py 与 convert_bag_to_lerobot.py）：
  - observation.state / action 均为 20 维：
      [左臂 joint_la0..la7 (8), 右臂 joint_ra0..ra7 (8),
       左夹爪, 右夹爪, head_pan, head_tilt]
  - 手臂/头部用增量(delta)，左右夹爪用绝对值（make_bool_mask(8,8,-1,-1,2)）
  - 头部无 state 话题，观测里 head_pan/head_tilt 恒填 0（与转换脚本一致）
  - 夹爪取值为归一化 [0,1]（mh2_const: FULLY_OPEN=[0,0], FULLY_CLOSE=[1,1]）
  - 图像 3 路：stereo_left_eye / stereo_right_eye / right_hand
      /stereo/left_eye/image_raw、/stereo/right_eye/image_raw、
      /right_hand/right_hand/color/image_raw
    部署端按训练转换脚本把图像拉伸到 224x224 RGB（模型内部会再 resize_with_pad，
    输入已是 224 时为不变换，从而与训练分布一致）

前置（工控机上先启动机器人 ROS 栈与相机）：
    # 关节/头部/夹爪控制器 + 相机（具体 bringup 以现场为准）
    ros2 launch ... （双臂/头部/夹爪/相机）
    # 确认话题有数据
    ros2 topic hz /joint_states
    ros2 topic hz /stereo/left_eye/image_raw

用法（工控机客户端：系统 python3 + ROS 环境，勿 activate venv；
--serve 本地推理：在 x86 GPU 服务器的 openpi venv 中运行）：
    source /opt/ros/humble/setup.bash
    source <机器人 ROS 工作空间>/install/setup.bash

  # ===== WebSocket 远程推理（推荐）=====
  # 只推理不运动（dry-run）
  python3 llbot_client/llbot_offline_deploy.py \
      --prompt "把红色方块放进篮子里" \
      --ws-host <server-ip> --ws-port 8000 \
      --frames 50 --control-hz 10

  # 真机运动（加 --move）
  python3 llbot_client/llbot_offline_deploy.py \
      --prompt "把红色方块放进篮子里" \
      --ws-host <server-ip> --ws-port 8000 \
      --replan-steps 10 --control-hz 10 --move

  # ===== 本地推理（推荐：常驻服务模式）=====
  # 终端1：启动模型服务器（加载模型可能数分钟，之后常驻）
  python3 llbot_client/llbot_offline_deploy.py --serve
  # 终端2：连接本地服务器推理（秒级响应）
  python3 llbot_client/llbot_offline_deploy.py \
      --prompt "把红色方块放进篮子里" \
      --ws-host localhost --ws-port 8001 --move

  # ===== 本地推理（直连，每次启动等加载，备用）=====
  python3 llbot_client/llbot_offline_deploy.py \
      --prompt "把红色方块放进篮子里" \
      --checkpoint-dir checkpoints/pi05_llbot_lora \
      --config-name pi05_llbot_lora \
      --frames 200 --control-hz 10 --move
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llbot_offline")

OPENPI_ROOT = Path(__file__).resolve().parent.parent

# ======================== 与训练数据一致的常量 ========================

# 手臂关节（/joint_states 中命名），顺序即训练 state 前 16 维
LEFT_ARM_JOINTS = [f"joint_la{i}" for i in range(8)]
RIGHT_ARM_JOINTS = [f"joint_ra{i}" for i in range(8)]
ARM_JOINT_ORDER = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS      # 16 维

# 头部关节（仅用于命令；state 恒为 0，因无头部 state 话题）
HEAD_JOINTS = ["joint_head0", "joint_head1"]

# 20 维 state/action 的名称（仅用于日志）
STATE_NAMES = (
    [f"left_arm_joint_{i}" for i in range(8)]
    + [f"right_arm_joint_{i}" for i in range(8)]
    + ["left_gripper", "right_gripper", "head_pan", "head_tilt"]
)

N_ARM = 16
N_STATE = 20
ACTION_DIM = 20

# 相机话题（键名必须与 llbot_policy 期望一致）
CAMERA_TOPICS = {
    "stereo_left_eye": "/stereo/left_eye/image_raw",
    "stereo_right_eye": "/stereo/right_eye/image_raw",
    "right_hand": "/right_hand/right_hand/color/image_raw",
}

JOINT_STATES_TOPIC = "/joint_states"
GRIPPER_STATE_TOPICS = {
    "left": "/left_gripper/joint_states",
    "right": "/right_gripper/joint_states",
}

# 命令话题
ARM_COMMAND_TOPIC = "/dual_arm_forward_position_controller/commands"   # Float64MultiArray, 16
HEAD_COMMAND_TOPIC = "/head_forward_position_controller/commands"      # Float64MultiArray, 2
GRIPPER_COMMAND_TOPICS = {
    "left": "/left_gripper/gripper_command",    # unix_msgs/GripperCommand
    "right": "/right_gripper/gripper_command",
}

# 夹爪：归一化 [0,1]，0=张开，1=闭合（mh2_const.HandConfig；URDF joint_lf 限位 [0,1]）
GRIPPER_MODE = "parallel"
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
GRIPPER_JOINT_NAMES = {"left": "joint_lf", "right": "joint_rf"}
# GripperCommand.command 的 name/velocity/effort 与采集端（遥操作）完全一致，
# 保证部署时下发的消息形态与训练数据来源一致
GRIPPER_CMD_NAMES = {"left": "gripper_left", "right": "gripper_right"}
GRIPPER_CMD_VELOCITY = 1.0
GRIPPER_CMD_EFFORT = 0.5

# 头部范围（弧度，mh2_const.HEAD_MIN/HEAD_MAX）
HEAD_MIN = np.array([-1.57, -0.2], dtype=np.float32)
HEAD_MAX = np.array([1.57, 0.35], dtype=np.float32)

# 图像目标尺寸（与转换脚本 TARGET_SIZE 一致，直接拉伸、不做 letterbox）
IMAGE_SIZE = (224, 224)  # (width, height)


def clip_gripper(v: float) -> float:
    """夹爪归一化值裁剪到合法范围 [0, 1]。"""
    return float(max(GRIPPER_MIN, min(GRIPPER_MAX, float(v))))


def clip_head(vec: np.ndarray) -> np.ndarray:
    """头部 2 维裁剪到合法角度范围（弧度）。"""
    return np.clip(np.asarray(vec, dtype=np.float32), HEAD_MIN, HEAD_MAX)


class ActionSmoother:
    """平滑动作块衔接，把「回抽」变成「减速平滑过去」（沿用 haibot_client 思路）

    本机器人动作 20 维：0:16 为双臂关节，16:18 为左右夹爪，18:20 为头部。
    平滑只作用于双臂（0:16）：夹爪要果断开合、头部行程小，均直通不参与混合/限速。

    两层处理：
    1. 混合过渡（splice_with_blend）：新块接入时不硬切，用升余弦权重把
       「旧计划的延续」与「新块」在 blend_window 步内加权过渡，把跳变摊到多步。
    2. 速度门控（gate）：方向阻尼（只压回抽分量）+ 逐关节限速（跳变兜底）。

    用法：
        smoother = ActionSmoother(max_vel_rad=0.12, blend_window=5)
        merged = smoother.splice_with_blend(action_queue, actions, start, last_cmd)
        action_queue.clear(); action_queue.extend(merged)
        cmd = smoother.gate(action, last_cmd); robot.send_action(cmd); last_cmd = cmd
    """

    def __init__(
        self,
        max_vel_rad: float = 0.12,
        blend_window: int = 5,
        back_damp: float = 0.15,
        release_steps: int = 5,
        vel_deadband: float = 0.004,
        n_arm: int = N_ARM,
    ):
        self.n_arm = int(n_arm)
        self.max_vel_rad = float(max_vel_rad)
        self.blend_window = int(blend_window)
        self.back_damp = float(np.clip(back_damp, 0.0, 1.0))
        self.release_steps = max(1, int(release_steps))
        self.vel_deadband = float(vel_deadband)

        self._vel_ema = np.zeros(self.n_arm, dtype=np.float32)
        self._vel_alpha = 0.5
        self._back_streak = 0

        self._backtrack_count = 0
        self._gate_count = 0
        self._blend_count = 0
        self._damped_rad: list[float] = []
        self._splice_lag: list[float] = []

    @staticmethod
    def _clip_tail(action: np.ndarray) -> None:
        """夹爪 [0,1] 裁剪，头部裁剪到角度范围（原地修改）。"""
        action[N_ARM] = clip_gripper(action[N_ARM])
        action[N_ARM + 1] = clip_gripper(action[N_ARM + 1])
        action[N_ARM + 2:N_ARM + 4] = clip_head(action[N_ARM + 2:N_ARM + 4])

    def gate(self, action: np.ndarray, last_cmd: Optional[np.ndarray]) -> np.ndarray:
        """速度门控：方向阻尼 + 逐关节限速 + 尾部分量裁剪"""
        action = np.asarray(action, dtype=np.float32).copy()
        if last_cmd is None:
            self._clip_tail(action)
            return action
        last_cmd = np.asarray(last_cmd, dtype=np.float32)

        delta = action[:self.n_arm] - last_cmd[:self.n_arm]

        # a) 方向阻尼：只压负向（回抽）分量，垂直分量（横向纠偏）完整保留
        damped = False
        released = False
        speed = float(np.linalg.norm(self._vel_ema))
        if self.back_damp < 1.0 and speed > self.vel_deadband:
            u = self._vel_ema / speed
            proj = float(np.dot(delta, u))
            if proj < 0.0:
                self._back_streak += 1
                if self._back_streak > self.release_steps:
                    k = min(1.0, (self._back_streak - self.release_steps) / self.release_steps)
                    damp = self.back_damp + (1.0 - self.back_damp) * k
                    released = damp >= 0.999
                else:
                    damp = self.back_damp
                if damp < 1.0:
                    delta = delta - (1.0 - damp) * proj * u
                    self._backtrack_count += 1
                    self._damped_rad.append(abs(proj) * (1.0 - damp))
                    damped = True
            else:
                self._back_streak = 0
        else:
            self._back_streak = 0

        # b) 逐关节限速：跳变兜底
        max_raw = float(np.abs(delta).max()) if delta.size else 0.0
        if max_raw > self.max_vel_rad:
            delta = np.clip(delta, -self.max_vel_rad, self.max_vel_rad)
            self._gate_count += 1
            logger.debug("[SMOOTH-GATE] 限速 %.3f -> %.3f rad/step (第%d次)",
                         max_raw, self.max_vel_rad, self._gate_count)

        action[:self.n_arm] = last_cmd[:self.n_arm] + delta
        self._clip_tail(action)

        # 运动方向 EMA：只用「真实推进」的步更新
        if not damped or released:
            eff = action[:self.n_arm] - last_cmd[:self.n_arm]
            self._vel_ema = ((1.0 - self._vel_alpha) * self._vel_ema
                             + self._vel_alpha * eff).astype(np.float32)
        return action

    def _extrapolate(self, prev: np.ndarray, vel: np.ndarray, steps: int) -> list:
        """旧队列不够长时，按上一段速度外推（带衰减）作为「旧计划的延续」"""
        out = []
        cur = np.asarray(prev, dtype=np.float32).copy()
        v = np.asarray(vel, dtype=np.float32).copy()
        for _ in range(steps):
            cur = cur.copy()
            cur[:self.n_arm] = cur[:self.n_arm] + v
            out.append(cur)
            v = (v * 0.7).astype(np.float32)
        return out

    def splice_with_blend(
        self,
        old_queue: Deque[np.ndarray],
        new_actions: np.ndarray,
        start_idx: int,
        last_cmd: Optional[np.ndarray] = None,
        end_idx: Optional[int] = None,
    ) -> list[np.ndarray]:
        """混合过渡：新序列接入时在双臂维度上创建平滑混合区（夹爪/头部直通）。"""
        new_actions = np.asarray(new_actions, dtype=np.float32)
        if new_actions.ndim == 1:
            new_actions = new_actions.reshape(1, -1)

        n_new = len(new_actions)
        stop = n_new if end_idx is None else min(int(end_idx), n_new)
        start_idx = max(0, int(start_idx))
        if start_idx >= stop:
            logger.warning("[SMOOTH-BLEND] 新序列已全部过期 (start=%d >= stop=%d)",
                           start_idx, stop)
            return []

        new_seq = [new_actions[i].copy() for i in range(start_idx, stop)]

        lag = 0.0
        speed = float(np.linalg.norm(self._vel_ema))
        if last_cmd is not None:
            last_cmd = np.asarray(last_cmd, dtype=np.float32)
            d0 = new_seq[0][:self.n_arm] - last_cmd[:self.n_arm]
            if speed > self.vel_deadband:
                proj = float(np.dot(d0, self._vel_ema / speed))
                lag = -proj if proj < 0.0 else 0.0
            self._splice_lag.append(lag)

        win = min(self.blend_window, len(new_seq))
        if win <= 1:
            logger.debug("[SMOOTH-BLEND] 混合窗口不足(win=%d)，直接接入", win)
            return new_seq

        old_list = list(old_queue)
        if len(old_list) >= win:
            ref = old_list[:win]
        else:
            if old_list:
                if len(old_list) >= 2:
                    vel = (np.asarray(old_list[-1], dtype=np.float32)[:self.n_arm]
                           - np.asarray(old_list[-2], dtype=np.float32)[:self.n_arm])
                elif last_cmd is not None:
                    vel = (np.asarray(old_list[0], dtype=np.float32)[:self.n_arm]
                           - last_cmd[:self.n_arm])
                else:
                    vel = self._vel_ema
                ref = list(old_list) + self._extrapolate(old_list[-1], vel, win - len(old_list))
            elif last_cmd is not None:
                ref = self._extrapolate(last_cmd, self._vel_ema, win)
            else:
                return new_seq

        out = []
        for i in range(win):
            t = (i + 1) / (win + 1)
            w = 0.5 - 0.5 * np.cos(np.pi * t)  # raised cosine: 0 → 1
            old_a = ref[i]
            new_a = new_seq[i].copy()
            # 仅双臂混合，夹爪/头部取新值
            new_a[:self.n_arm] = (1.0 - w) * old_a[:self.n_arm] + w * new_a[:self.n_arm]
            out.append(new_a)
        out.extend(new_seq[win:])

        self._blend_count += 1
        logger.info("[SMOOTH-BLEND] 混合%d步 (接入点滞后=%.4f rad) 第%d次",
                    win, lag, self._blend_count)
        return out

    def get_stats(self) -> dict:
        st = {
            "backtrack_count": self._backtrack_count,
            "gate_count": self._gate_count,
            "blend_count": self._blend_count,
        }
        if self._damped_rad:
            st["avg_damped_rad"] = float(np.mean(self._damped_rad))
            st["max_damped_rad"] = float(np.max(self._damped_rad))
        if self._splice_lag:
            st["avg_splice_lag"] = float(np.mean(self._splice_lag))
            st["max_splice_lag"] = float(np.max(self._splice_lag))
        return st


def imgmsg_to_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image → HxWx3 uint8 RGB"""
    from cv_bridge import CvBridge

    bridge = CvBridge()
    return bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")


def preprocess_image(rgb: np.ndarray) -> np.ndarray:
    """RGB HxWx3 → 224x224x3 uint8 RGB（与转换脚本一致：直接拉伸、PIL BILINEAR）"""
    from PIL import Image

    img = np.asarray(rgb)
    if img.ndim == 3 and img.shape[0] == 3:  # CHW → HWC 兜底
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    pil = Image.fromarray(img).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


class RobotIO:
    """订阅相机/关节/夹爪，发布双臂 + 头部 + 夹爪命令"""

    def __init__(self, node, enable_move: bool):
        from rclpy.qos import (
            HistoryPolicy,
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Float64MultiArray
        from unix_msgs.msg import GripperCommand

        self._node = node
        self._enable_move = enable_move
        self._lock = threading.Lock()
        self._latest_images: Dict[str, np.ndarray] = {}
        self._latest_joint: Optional[Dict[str, float]] = None
        self._latest_gripper: Dict[str, float] = {}
        self._joint_msg_count = 0
        self._gripper_msg_count = 0
        self._image_msg_count = {k: 0 for k in CAMERA_TOPICS}

        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        for name, topic in CAMERA_TOPICS.items():
            node.create_subscription(
                Image, topic, lambda msg, n=name: self._on_image(n, msg), img_qos)
            logger.info("订阅相机: %s", topic)

        node.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint, 10)
        logger.info("订阅关节: %s", JOINT_STATES_TOPIC)

        for side, topic in GRIPPER_STATE_TOPICS.items():
            node.create_subscription(
                JointState, topic, lambda msg, s=side: self._on_gripper(s, msg), 10)
            logger.info("订阅夹爪: %s", topic)

        self._arm_pub = node.create_publisher(Float64MultiArray, ARM_COMMAND_TOPIC, 10)
        self._head_pub = node.create_publisher(Float64MultiArray, HEAD_COMMAND_TOPIC, 10)
        self._gripper_pub: Dict[str, object] = {}
        if enable_move:
            for side, topic in GRIPPER_COMMAND_TOPICS.items():
                self._gripper_pub[side] = node.create_publisher(GripperCommand, topic, 10)
                logger.info("夹爪下发: %s", topic)

        self._Float64MultiArray = Float64MultiArray
        self._JointState = JointState
        self._GripperCommand = GripperCommand

    def _on_image(self, name: str, msg) -> None:
        try:
            img = imgmsg_to_rgb(msg)
        except Exception as exc:
            logger.warning("图像转换失败 [%s]: %s", name, exc)
            return
        with self._lock:
            self._latest_images[name] = img
            self._image_msg_count[name] += 1

    def _on_joint(self, msg) -> None:
        mapping = {n: float(p) for n, p in zip(msg.name, msg.position)}
        with self._lock:
            self._latest_joint = mapping
            self._joint_msg_count += 1

    def _on_gripper(self, side: str, msg) -> None:
        if msg.position:
            with self._lock:
                self._latest_gripper[side] = float(msg.position[0])
                self._gripper_msg_count += 1

    def wait_ready(self, timeout_s: float = 30.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._lock:
                cams_ok = all(k in self._latest_images for k in CAMERA_TOPICS)
                joints_ok = self._latest_joint is not None
                gripper_ok = all(s in self._latest_gripper for s in GRIPPER_STATE_TOPICS)
            if cams_ok and joints_ok and gripper_ok:
                logger.info("传感器就绪: 关节+双夹爪+三路相机")
                return
            time.sleep(0.1)
        missing = []
        with self._lock:
            for k in CAMERA_TOPICS:
                if k not in self._latest_images:
                    missing.append(CAMERA_TOPICS[k])
            if self._latest_joint is None:
                missing.append(JOINT_STATES_TOPIC)
            for s, t in GRIPPER_STATE_TOPICS.items():
                if s not in self._latest_gripper:
                    missing.append(t)
        raise TimeoutError(f"等待传感器超时，缺少: {missing}")

    def get_state(self) -> np.ndarray:
        """构造 20 维 state（头部 state 恒为 0，与训练转换脚本一致）"""
        with self._lock:
            if self._latest_joint is None:
                raise RuntimeError("尚未收到 joint_states")
            arm_mapping = dict(self._latest_joint)
            grip_mapping = dict(self._latest_gripper)

        state = np.zeros(N_STATE, dtype=np.float32)
        missing = [n for n in ARM_JOINT_ORDER if n not in arm_mapping]
        if missing:
            raise KeyError(
                f"/joint_states 缺少手臂关节 {missing}；"
                f"实际关节名={list(arm_mapping)}。请确认命名为 joint_la0-7 / joint_ra0-7")
        for i, name in enumerate(ARM_JOINT_ORDER):
            state[i] = arm_mapping[name]

        for side, idx in (("left", N_ARM), ("right", N_ARM + 1)):
            if side not in grip_mapping:
                raise KeyError(f"{GRIPPER_STATE_TOPICS[side]} 尚未收到夹爪状态")
            state[idx] = grip_mapping[side]

        # head_pan / head_tilt：无 state 话题，用 0 填充
        state[N_ARM + 2:N_ARM + 4] = 0.0
        return state

    def get_images(self) -> Dict[str, np.ndarray]:
        with self._lock:
            missing = [k for k in CAMERA_TOPICS if k not in self._latest_images]
            if missing:
                raise RuntimeError(f"缺少相机帧: {missing}")
            raw = {k: self._latest_images[k].copy() for k in CAMERA_TOPICS}
        return {k: preprocess_image(v) for k, v in raw.items()}

    def send_action(self, action: np.ndarray) -> None:
        """下发 20 维动作：双臂(16) + 头部(2) + 夹爪(2)"""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < ACTION_DIM:
            raise ValueError(f"动作维度至少 {ACTION_DIM}, 得到 {action.shape}")

        if not self._enable_move:
            return

        arm_msg = self._Float64MultiArray()
        arm_msg.data = action[:N_ARM].astype(np.float64).tolist()
        self._arm_pub.publish(arm_msg)

        head_msg = self._Float64MultiArray()
        head_msg.data = clip_head(action[N_ARM + 2:N_ARM + 4]).astype(np.float64).tolist()
        self._head_pub.publish(head_msg)

        # 外走线夹爪：GripperCommand(mode=parallel, command.position=[归一化 0..1])
        for side, idx in (("left", N_ARM), ("right", N_ARM + 1)):
            pub = self._gripper_pub.get(side)
            if pub is None:
                continue
            msg = self._GripperCommand()
            msg.mode = GRIPPER_MODE
            js = self._JointState()
            js.name = [GRIPPER_CMD_NAMES[side]]
            js.position = [clip_gripper(float(action[idx]))]
            js.velocity = [GRIPPER_CMD_VELOCITY]
            js.effort = [GRIPPER_CMD_EFFORT]
            msg.command = js
            pub.publish(msg)


def load_policy(checkpoint_dir: str, config_name: str, device: str, prompt: Optional[str]):
    from openpi.policies import policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(config_name)
    logger.info("加载模型 config=%s checkpoint=%s compile=%s",
                config_name, checkpoint_dir, train_config.model.pytorch_compile_mode)
    t0 = time.time()
    policy = policy_config.create_trained_policy(
        train_config, checkpoint_dir, pytorch_device=device, default_prompt=prompt)
    logger.info("模型加载完成 (%.1fs)", time.time() - t0)
    return policy


def _ts() -> str:
    return time.strftime("%H:%M:%S.") + f"{int((time.time() % 1) * 1000):03d}"


def _log_infer(step: int, state: np.ndarray, actions: np.ndarray, infer_ms: float,
               n_take: int, latency_ms: float = 0.0) -> None:
    sl = " ".join(f"{v:.4f}" for v in state[:8])
    sr = " ".join(f"{v:.4f}" for v in state[8:16])
    head = (float(state[18]), float(state[19]))
    tag = f"infer={infer_ms:.1f}ms latency={latency_ms:.1f}ms" if latency_ms > 0 \
        else f"infer={infer_ms:.1f}ms"
    logger.info("推理#%d %s take=%d/%d | L=[%s] R=[%s] grip=(%.3f,%.3f) head=(%.3f,%.3f)",
                step, tag, n_take, actions.shape[0], sl, sr,
                float(state[16]), float(state[17]), head[0], head[1])
    for i in range(min(n_take, actions.shape[0])):
        ai = actions[i]
        logger.info("  step%d: L=[%s] R=[%s] grip=(%.3f,%.3f) head=(%.3f,%.3f)",
                    i,
                    " ".join(f"{v:.4f}" for v in ai[:8]),
                    " ".join(f"{v:.4f}" for v in ai[8:16]),
                    float(ai[16]), float(ai[17]), float(ai[18]), float(ai[19]))


def run_loop(
    policy,
    robot: RobotIO,
    prompt: str,
    frames: int,
    control_hz: float,
    replan_steps: int,
    use_async: bool = True,
    smoother: Optional[ActionSmoother] = None,
) -> None:
    """推理-控制循环（异步推理 + 防回抽平滑）

    use_async=True（默认）：推理在后台线程执行，队列剩余 ≤ 阈值时提前触发，
    队列耗尽时 hold 最后动作；触发阈值按首次推理延迟固定。
    use_async=False（同步）：队列耗尽才推理，推理期间机器人停顿。
    """
    period = 1.0 / max(control_hz, 1e-3)
    action_queue: Deque[np.ndarray] = deque()
    last_infer_ms = 0.0
    infinite = frames <= 0
    max_steps = frames if not infinite else float("inf")
    last_send_t: Optional[float] = None
    stall_ms_list: list[float] = []

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
    pending_infer: Optional[Future] = None
    infer_count = 0
    infer_trigger_step = 0
    last_cmd: Optional[np.ndarray] = None
    fallback_count = 0
    trigger_threshold: Optional[int] = None
    total_latency_ms = 0.0

    def _async_infer_task(obs: dict, infer_id: int) -> dict:
        t0 = time.time()
        try:
            result = policy.infer(obs)
            latency_ms = (time.time() - t0) * 1000.0
            return {"success": True, "result": result, "latency_ms": latency_ms,
                    "infer_id": infer_id}
        except Exception as e:
            logger.error("[ASYNC] 推理失败 id=%d: %s", infer_id, e)
            return {"success": False, "error": str(e), "infer_id": infer_id}

    def _trigger_infer(step: int) -> None:
        nonlocal pending_infer, infer_count, infer_trigger_step
        if pending_infer is not None and not pending_infer.done():
            return
        infer_count += 1
        infer_trigger_step = step
        t0 = time.time()
        state = robot.get_state()          # 头部 state 已在 get_state 里置 0
        images = robot.get_images()        # 已做 224x224 RGB 预处理
        observation = {"state": state, "images": images, "prompt": prompt}
        obs_ms = (time.time() - t0) * 1000.0
        logger.info("[ASYNC] 触发推理#%d ts=%s step=%d queue=%d obs_prep=%.1fms",
                    infer_count, _ts(), step, len(action_queue), obs_ms)
        pending_infer = executor.submit(_async_infer_task, observation, infer_count)
        if not use_async:
            pending_infer.result()
            _check_infer_result(step)

    def _check_infer_result(step: int) -> bool:
        nonlocal pending_infer, last_infer_ms, trigger_threshold, total_latency_ms
        if pending_infer is None or not pending_infer.done():
            return False

        result_dict = pending_infer.result()
        pending_infer = None
        if not result_dict["success"]:
            logger.warning("[ASYNC] 推理失败，跳过: %s", result_dict.get("error"))
            return False

        latency_ms = result_dict["latency_ms"]
        result = result_dict["result"]
        actions = np.asarray(result["actions"])
        last_infer_ms = float(result.get("policy_timing", {}).get("infer_ms", 0.0))
        n_take = min(replan_steps, actions.shape[0])

        total_latency_ms += latency_ms
        if trigger_threshold is None:
            trigger_threshold = max(1, int(latency_ms / (period * 1000.0)) + 2)
            logger.info("[ASYNC] 固定触发阈值=%d步 (首次latency=%.1fms, 周期=%.1fms)",
                        trigger_threshold, latency_ms, period * 1000.0)

        consumed = max(0, step - infer_trigger_step)
        discarded = len(action_queue)
        start = min(consumed, n_take)

        if smoother is not None:
            merged = smoother.splice_with_blend(action_queue, actions, start, last_cmd)
            action_queue.clear()
            action_queue.extend(merged)
            n_new = len(merged)
        else:
            action_queue.clear()
            for i in range(start, n_take):
                action_queue.append(actions[i])
            n_new = n_take - start

        if consumed >= n_take and len(action_queue) == 0:
            logger.warning("[ASYNC] 推理#%d完成但已全部过期(consumed=%d>=take=%d)，队列空，立即重推",
                           result_dict["infer_id"], consumed, n_take)

        lag_info = ""
        if smoother is not None and smoother._splice_lag:
            lag_info = f" splice_lag={smoother._splice_lag[-1]:.4f}rad"

        logger.info(
            "[ASYNC] 推理#%d完成 ts=%s latency=%.1fms server_infer=%.1fms "
            "consumed=%d skip=%d 丢弃旧动作%d步 新入队%d步 queue=%d threshold=%d%s",
            result_dict["infer_id"], _ts(), latency_ms, last_infer_ms,
            consumed, start, discarded, n_new, len(action_queue), trigger_threshold, lag_info)
        return True

    def _get_fallback_action() -> np.ndarray:
        nonlocal fallback_count
        fallback_count += 1
        if last_cmd is not None:
            logger.warning("[FALLBACK] 队列耗尽#%d，hold最后动作 (推理中=%s)",
                           fallback_count, pending_infer is not None and not pending_infer.done())
            return last_cmd.copy()
        logger.error("[FALLBACK] 队列耗尽且无历史动作，保持当前位置")
        return robot.get_state()

    logger.info(
        "开始%s控制循环: frames=%s, control_hz=%.1f, replan_steps=%d, prompt=%r, smoother=%s",
        "异步推理" if use_async else "同步推理",
        "无限" if infinite else frames, control_hz, replan_steps, prompt,
        "启用" if smoother is not None else "禁用")

    # 初始推理
    logger.info("[INIT] 启动初始推理...")
    if smoother is not None:
        last_cmd = robot.get_state()
        logger.info("[INIT] last_cmd锚定到当前状态（头部2维为0，仅双臂参与平滑）")
    _trigger_infer(0)
    while pending_infer is not None and not pending_infer.done():
        time.sleep(0.01)
    _check_infer_result(0)
    if len(action_queue) == 0:
        raise RuntimeError("初始推理失败，队列为空")
    logger.info("[INIT] 初始推理完成，队列长度=%d，进入控制循环", len(action_queue))

    step = 0
    try:
        while infinite or step < max_steps:
            t_start = time.time()

            got_new = _check_infer_result(step)

            if use_async:
                if len(action_queue) <= trigger_threshold:
                    if pending_infer is None or pending_infer.done():
                        _trigger_infer(step)
            else:
                if len(action_queue) == 0:
                    _trigger_infer(step)

            if len(action_queue) > 0:
                action = action_queue.popleft()
            else:
                action = _get_fallback_action()

            if smoother is not None:
                action = smoother.gate(action, last_cmd)

            t_send = time.time()
            robot.send_action(action)
            last_cmd = np.asarray(action, dtype=np.float32).copy()

            had_prev = last_send_t is not None
            gap_ms = ((t_send - last_send_t) * 1000.0) if had_prev else 0.0
            stall_ms = max(0.0, gap_ms - period * 1000.0) if had_prev else 0.0
            if had_prev and stall_ms > 5.0:
                stall_ms_list.append(stall_ms)
            last_send_t = t_send
            step += 1

            if got_new or step % max(1, int(control_hz)) == 0 or stall_ms > period * 500.0:
                logger.info("[STEP] ts=%s step=%d queue=%d gap_ms=%.1f stall_ms=%.1f%s%s",
                            _ts(), step, len(action_queue), gap_ms, stall_ms,
                            " ← 新推理入队" if got_new else "",
                            " ← 卡顿" if stall_ms > period * 500.0 else "")

            elapsed = time.time() - t_start
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        executor.shutdown(wait=False)
        if stall_ms_list:
            arr = np.asarray(stall_ms_list, dtype=np.float64)
            logger.info("[TIMING] 卡顿统计(次=%d): mean=%.1fms p50=%.1fms p95=%.1fms max=%.1fms",
                        len(arr), float(arr.mean()), float(np.percentile(arr, 50)),
                        float(np.percentile(arr, 95)), float(arr.max()))
        if fallback_count > 0:
            logger.warning("[SUMMARY] Fallback触发次数: %d", fallback_count)
        if smoother is not None:
            logger.info("[SUMMARY] 平滑器统计: %s", smoother.get_stats())
        logger.info("[SUMMARY] 控制循环结束: 总步数=%d, 推理次数=%d, 平均推理延迟=%.1fms",
                    step, infer_count, (total_latency_ms / infer_count) if infer_count else 0.0)


def main():
    parser = argparse.ArgumentParser(description="LLbot 工控机离线真机部署")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(OPENPI_ROOT / "checkpoints" / "pi05_llbot_lora"),
                        help="PyTorch 检查点目录")
    parser.add_argument("--config-name", type=str, default="pi05_llbot_lora", help="训练配置名")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--prompt", type=str, default=None, help="语言指令（--serve 模式可省略）")
    parser.add_argument("--ws-host", type=str, default=None,
                        help="WebSocket 推理服务器地址（指定则使用远程推理）")
    parser.add_argument("--ws-port", type=int, default=8000, help="WebSocket 推理服务器端口")
    parser.add_argument("--frames", type=int, default=0, help="控制步数，0=无限循环 Ctrl+C 停止")
    parser.add_argument("--control-hz", type=float, default=10.0, help="控制频率 Hz")
    parser.add_argument("--replan-steps", type=int, default=1,
                        help="每次推理后取前 N 步动作再重新推理（默认 1）")
    parser.add_argument("--move", action="store_true",
                        help="真正下发关节/头部/夹爪指令；不加则只推理不运动（dry-run）")
    parser.add_argument("--sync", action="store_true",
                        help="同步推理（串行）：队列耗尽才推理，推理期间停顿（默认异步）")
    parser.add_argument("--wait-timeout", type=float, default=30.0, help="等待相机/关节超时秒")
    parser.add_argument("--serve", type=int, nargs="?", const=8001, default=None, metavar="PORT",
                        help="常驻模式：加载模型后开启 WebSocket 服务器（默认端口 8001）")

    # 平滑参数（默认启用）
    parser.add_argument("--no-smooth", action="store_true", help="禁用防回抽平滑（默认启用）")
    parser.add_argument("--max-vel", type=float, default=0.12,
                        help="平滑器单关节每步最大变化 rad/step（默认 0.12≈1.2rad/s@10Hz）")
    parser.add_argument("--blend-steps", type=int, default=5,
                        help="平滑器混合窗口步数（默认 5），0=关闭混合")
    parser.add_argument("--back-damp", type=float, default=0.15,
                        help="平滑器回抽方向保留比例（默认 0.15），0=完全不许回抽")
    args = parser.parse_args()

    if args.prompt is None and args.serve is None:
        raise SystemExit("缺少 --prompt（--serve 常驻模式可省略）")

    if not args.move:
        logger.warning("当前为 dry-run（未加 --move），只推理不发指令")

    # 阶段1：准备 policy
    if args.ws_host:
        logger.info("===== 阶段1: 连接 WebSocket 推理服务器 %s:%s =====",
                    args.ws_host, args.ws_port)
        from openpi_client import websocket_client_policy
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.ws_host, port=args.ws_port)
    else:
        logger.info("===== 阶段1: 加载本地模型（可能数分钟，勿杀进程）=====")
        policy = load_policy(args.checkpoint_dir, args.config_name, args.device, args.prompt)

    if args.serve is not None:
        logger.info("===== 阶段2: 常驻服务模式（端口 %d）=====", args.serve)
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        server = WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.serve)
        logger.info("模型服务器就绪，等待客户端连接...")
        server.serve_forever()
        return

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    logger.info("===== 阶段2: 启动 ROS，等待关节+双夹爪+三路相机 =====")
    rclpy.init()
    node = Node("llbot_offline_deploy")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = RobotIO(node, enable_move=args.move)
    try:
        robot.wait_ready(timeout_s=args.wait_timeout)

        smoother = None
        if not args.no_smooth:
            smoother = ActionSmoother(
                max_vel_rad=args.max_vel,
                blend_window=args.blend_steps,
                back_damp=args.back_damp,
            )
            logger.info("防回抽平滑已启用 (max_vel=%.3f, blend=%d, back_damp=%.2f)",
                        args.max_vel, args.blend_steps, args.back_damp)
        else:
            logger.info("防回抽平滑已禁用 (--no-smooth)")

        logger.info("===== 阶段3: 真机实时控制循环 =====")
        run_loop(
            policy=policy,
            robot=robot,
            prompt=args.prompt,
            frames=args.frames,
            control_hz=args.control_hz,
            replan_steps=args.replan_steps,
            use_async=not args.sync,
            smoother=smoother,
        )
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
