"""LLbot机器人策略适配

数据映射（与 LeRobot 数据集 meta/info.json 一致）：
- 观测状态 20维: left_arm_joint_0~7, right_arm_joint_0~7,
  left_gripper, right_gripper, head_pan, head_tilt
- 动作 20维: 与状态同维度同顺序
- 图像: stereo_left_eye, stereo_right_eye, right_hand

图像槽位映射（π0.5 仅支持 3 路图像）：
- stereo_left_eye  -> base_0_rgb        (双目左眼，第三人称主视角)
- stereo_right_eye -> left_wrist_0_rgb  (双目右眼)
- right_hand       -> right_wrist_0_rgb (右手相机)

动作 delta 掩码 make_bool_mask(8, 8, -1, -1, 2)：
- 左臂8关节、右臂8关节、头部2自由度 -> 增量 (delta)
- 左夹爪、右夹爪 -> 绝对值
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_llbot_example() -> dict:
    """创建LLbot机器人的随机输入示例"""
    return {
        "state": np.ones((20,)),
        "images": {
            "stereo_left_eye": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "stereo_right_eye": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_hand": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class LLbotInputs(transforms.DataTransformFn):
    """LLbot策略的输入转换"""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("stereo_left_eye", "stereo_right_eye", "right_hand")

    def __call__(self, data: dict) -> dict:
        data = _decode_llbot(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["stereo_left_eye"]

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": in_images.get("stereo_right_eye", np.zeros_like(base_image)),
            "right_wrist_0_rgb": in_images.get("right_hand", np.zeros_like(base_image)),
        }

        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_ if "stereo_right_eye" in in_images else np.False_,
            "right_wrist_0_rgb": np.True_ if "right_hand" in in_images else np.False_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LLbotOutputs(transforms.DataTransformFn):
    """LLbot策略的输出转换（推理时使用）"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :20])}


def _decode_llbot(data: dict) -> dict:
    """解码LLbot数据格式"""
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        if img.shape[0] == 3:
            img = einops.rearrange(img, "c h w -> h w c")
        return img

    images = data.get("images", {})
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data
