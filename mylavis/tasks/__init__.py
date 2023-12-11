"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from mylavis.common.registry import registry
from mylavis.tasks.base_task import BaseTask
from mylavis.tasks.captioning import CaptionTask
from mylavis.tasks.image_text_pretrain import ImageTextPretrainTask
from mylavis.tasks.multimodal_classification import (
    MultimodalClassificationTask,
)
from mylavis.tasks.retrieval import RetrievalTask
from mylavis.tasks.vqa import VQATask, AOKVQATask
from mylavis.tasks.vqa_reading_comprehension import VQARCTask, GQARCTask
from mylavis.tasks.dialogue import DialogueTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "AOKVQATask",
    "RetrievalTask",
    "CaptionTask",
    "VQATask",
    "VQARCTask",
    "GQARCTask",
    "MultimodalClassificationTask",
    # "VideoQATask",
    # "VisualEntailmentTask",
    "ImageTextPretrainTask",
    "DialogueTask",
]
