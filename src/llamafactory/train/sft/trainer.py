# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import shutil
import sys
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import yaml
from safetensors.torch import save_file as safe_save_file
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import _get_decay_parameter_names, create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _get_current_yaml_config() -> Optional[dict[str, Any]]:
    candidate_paths = []
    env_path = os.environ.get("FUNAUDIOCHAT_TRAIN_CONFIG")
    if env_path:
        candidate_paths.append(env_path)
    candidate_paths.extend(arg for arg in sys.argv[1:] if str(arg).endswith((".yaml", ".yml")))

    for candidate in candidate_paths:
        path = candidate if os.path.isabs(candidate) else os.path.abspath(candidate)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning_rank0(f"Failed to read FunAudioChat training config {path}: {exc}")
            return None
        return data if isinstance(data, dict) else None
    return None


def _format_param_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.3f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.3f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.3f}K"
    return str(num_params)


def _safe_clip_grad_norm_fp64(parameters, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    if norm_type != 2 and norm_type != 2.0:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    params = [p for p in parameters if p is not None and p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    device = params[0].grad.device
    total_sq = torch.zeros((), device=device, dtype=torch.float64)
    nonfinite = False
    for param in params:
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not bool(finite.all()):
            nonfinite = True
            grad = torch.where(finite, grad, torch.zeros_like(grad))
        total_sq += grad.double().pow(2).sum()

    total_norm = torch.sqrt(total_sq)
    if nonfinite:
        return torch.full((), float("nan"), device=device)

    max_norm = float(max_norm)
    if max_norm > 0.0 and bool(torch.isfinite(total_norm)) and float(total_norm.detach().cpu()) > max_norm:
        scale = max_norm / (float(total_norm.detach().cpu()) + 1.0e-12)
        for param in params:
            param.grad.detach().mul_(scale)
    return total_norm.to(device=device, dtype=torch.float32)


def _get_trainable_bucket(name: str) -> str:
    name = name.removeprefix("module.")
    is_lora = "lora_" in name or ".lora_" in name
    if name.startswith("audio_invert_tower."):
        return "audio_invert_lora" if is_lora else "audio_invert_tower"
    if is_lora:
        return "llm_lora"
    if name.startswith("listen_context_aggregator."):
        return "listen_context_aggregator"
    if name.startswith("listen_action_refinement_head."):
        return "listen_action_refinement_head"
    if name.startswith("speak_action_refinement_head."):
        return "speak_action_refinement_head"
    if name.startswith("action_refinement_head."):
        return "action_refinement_head"
    return "other_trainable"


def _get_funaudio_module_lrs() -> dict[str, float]:
    config = _get_current_yaml_config()
    if not config:
        return {}

    def _read_lr(block_name: str, flat_name: str) -> Optional[float]:
        value = config.get(flat_name)
        block = config.get(block_name)
        if value is None and isinstance(block, dict):
            value = block.get("learning_rate")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning_rank0(f"Ignoring invalid learning rate for {block_name}: {value}")
            return None

    lrs: dict[str, float] = {}
    lora_lr = _read_lr("lora", "lora_learning_rate")
    if lora_lr is None:
        lora_lr = _read_lr("lora_llm", "lora_llm_learning_rate")
    if lora_lr is not None:
        lrs["lora"] = lora_lr

    audio_invert_lr = _read_lr("audio_invert_tower", "audio_invert_tower_learning_rate")
    if audio_invert_lr is not None:
        lrs["audio_invert_tower"] = audio_invert_lr

    action_lr = _read_lr("action_refinement", "action_refinement_learning_rate")
    listen_action_lr = _read_lr("listen_action", "listen_action_learning_rate")
    if listen_action_lr is None:
        listen_action_lr = action_lr
    if listen_action_lr is not None:
        lrs["listen_action"] = listen_action_lr

    speak_action_lr = _read_lr("speak_action", "speak_action_learning_rate")
    if speak_action_lr is None:
        speak_action_lr = action_lr
    if speak_action_lr is not None:
        lrs["speak_action"] = speak_action_lr

    listen_semantic_lr = _read_lr("listen_semantic", "listen_semantic_learning_rate")
    if listen_semantic_lr is not None:
        lrs["listen_semantic"] = listen_semantic_lr

    speak_input_gate_lr = _read_lr("speak_input_gate", "speak_input_gate_learning_rate")
    if speak_input_gate_lr is not None:
        lrs["speak_input_gate"] = speak_input_gate_lr
    return lrs


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _get_funaudio_save_config() -> dict[str, Any]:
    config = _get_current_yaml_config() or {}
    return {
        "save_trained_modules_only": _as_bool(
            config.get("save_trained_modules_only", config.get("funaudio_save_trained_modules_only")),
            default=False,
        ),
        "trained_modules_dir": str(config.get("trained_modules_dir", "trained_modules")),
    }


_FUN_AUDIO_CONFIG_KEYS = [
    "enable_action_refinement_head",
    "action_vocab_size",
    "action_num_quantizers",
    "action_refinement_hidden_size",
    "action_refinement_num_heads",
    "action_refinement_num_layers",
    "action_refinement_dropout",
    "action_refinement_listen_hidden_size",
    "action_refinement_listen_num_heads",
    "action_refinement_listen_num_layers",
    "action_refinement_listen_dropout",
    "action_refinement_listen_context_enabled",
    "action_refinement_listen_context_num_heads",
    "action_refinement_listen_context_num_layers",
    "action_refinement_listen_context_dropout",
    "action_refinement_listen_context_audio_gate_init",
    "action_refinement_listen_rollout_horizon",
    "action_refinement_listen_rollout_loss_weight",
    "action_refinement_listen_rollout_batch_prob",
    "action_refinement_speak_hidden_size",
    "action_refinement_speak_num_heads",
    "action_refinement_speak_num_layers",
    "action_refinement_speak_dropout",
    "action_refinement_listen_arch",
    "action_refinement_listen_input_source",
    "action_refinement_route_for_listen",
    "action_refinement_route_for_speak",
    "action_refinement_default_mode",
    "action_refinement_separate_listen_speak",
    "action_refinement_use_prev_action",
    "action_refinement_enable_motion_buffer",
    "action_refinement_enable_adaln",
    "action_refinement_listen_zero_condition",
    "action_refinement_detach_backbone_for_listen",
    "action_refinement_detach_backbone_for_speak",
    "action_refinement_speak_backbone_grad_scale",
    "action_refinement_speak_use_text_embeds",
    "action_refinement_listen_fp32",
    "action_refinement_input_norm",
    "action_refinement_input_gate_init",
    "action_refinement_condition_gate_init",
    "action_refinement_speak_condition_layers",
    "action_refinement_xavier_init",
    "action_refinement_zero_init_adaln",
    "action_refinement_detach_condition_for_speak",
    "action_refinement_speak_condition_dropout",
    "action_refinement_label_mask",
    "action_refinement_loss_weight",
    "freeze_audio_modules",
    "unfreeze_audio_invert_tower",
    "audio_invert_tower_lora",
]


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        elif os.environ.get("FUNAUDIOCHAT_SAFE_GRAD_CLIP", "1").strip().lower() in {"1", "true", "yes", "y", "on"}:
            original_clip_grad_norm = self.accelerator.clip_grad_norm_

            def _clip_grad_norm_safe(accelerator, parameters, max_norm, norm_type=2):
                distributed_type = str(getattr(accelerator, "distributed_type", "")).lower()
                if "deepspeed" in distributed_type or "fsdp" in distributed_type or norm_type not in {2, 2.0}:
                    return original_clip_grad_norm(parameters, max_norm, norm_type=norm_type)
                accelerator.unscale_gradients()
                return _safe_clip_grad_norm_fp64(parameters, max_norm, norm_type=norm_type)

            self.accelerator.clip_grad_norm_ = MethodType(_clip_grad_norm_safe, self.accelerator)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        self._log_funaudio_training_summary()

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

        self._funaudio_loss_buffer = {
            "text_loss": [],
            "speech_loss": [],
            "listen_action_loss": [],
            "speak_action_loss": [],
        }
        self._funaudio_grad_buffer = {}
        if self._should_log_funaudio_grad_norms():
            self._funaudio_grad_buffer = {
                "listen_context_grad_norm": [],
                "listen_action_grad_norm": [],
                "speak_action_grad_norm": [],
                "action_grad_norm": [],
                "audio_invert_grad_norm": [],
                "llm_lora_grad_norm": [],
                "other_grad_norm": [],
                "listen_context_grad_nonfinite": [],
                "listen_action_grad_nonfinite": [],
                "speak_action_grad_nonfinite": [],
                "action_grad_nonfinite": [],
                "audio_invert_grad_nonfinite": [],
                "llm_lora_grad_nonfinite": [],
                "other_grad_nonfinite": [],
            }
    def _create_funaudio_module_lr_optimizer(self) -> Optional["torch.optim.Optimizer"]:
        module_lrs = _get_funaudio_module_lrs()
        if not module_lrs:
            return None

        base_lr = float(self.args.learning_rate)
        lora_lr = module_lrs.get("lora", base_lr)
        audio_invert_lr = module_lrs.get("audio_invert_tower", base_lr)
        listen_action_lr = module_lrs.get("listen_action", base_lr)
        speak_action_lr = module_lrs.get("speak_action", base_lr)
        listen_semantic_lr = module_lrs.get("listen_semantic", listen_action_lr)
        speak_input_gate_lr = module_lrs.get("speak_input_gate", speak_action_lr)
        decay_param_names = set(_get_decay_parameter_names(self.model))
        grouped_params: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
            ("base", True): [],
            ("base", False): [],
            ("lora", True): [],
            ("lora", False): [],
            ("audio_invert_tower", True): [],
            ("audio_invert_tower", False): [],
            ("listen_action", True): [],
            ("listen_action", False): [],
            ("speak_action", True): [],
            ("speak_action", False): [],
            ("listen_semantic", True): [],
            ("listen_semantic", False): [],
            ("speak_input_gate", True): [],
            ("speak_input_gate", False): [],
        }

        def _module_key(name: str) -> str:
            name = name.removeprefix("module.")
            if name.startswith("audio_invert_tower."):
                return "audio_invert_tower"
            if name.startswith((
                "listen_action_refinement_head.semantic_gate",
                "listen_action_refinement_head.semantic_cond_proj.",
            )):
                return "listen_semantic"
            if name == "speak_action_refinement_head.input_gate":
                return "speak_input_gate"
            if name.startswith((
                "action_refinement_head.",
                "listen_context_aggregator.",
                "listen_action_refinement_head.",
            )):
                return "listen_action"
            if name.startswith("speak_action_refinement_head."):
                return "speak_action"
            if "lora_" in name or ".lora_" in name:
                return "lora"
            return "base"

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            grouped_params[(_module_key(name), name in decay_param_names)].append(param)

        group_lrs = {
            "base": base_lr,
            "lora": lora_lr,
            "audio_invert_tower": audio_invert_lr,
            "listen_action": listen_action_lr,
            "speak_action": speak_action_lr,
            "listen_semantic": listen_semantic_lr,
            "speak_input_gate": speak_input_gate_lr,
        }
        param_groups = []
        param_counts: dict[str, int] = {key: 0 for key in group_lrs}
        for module_key, lr in group_lrs.items():
            for use_decay in (True, False):
                params = grouped_params[(module_key, use_decay)]
                if not params:
                    continue
                param_groups.append(
                    {
                        "params": params,
                        "lr": lr,
                        "weight_decay": self.args.weight_decay if use_decay else 0.0,
                    }
                )
                param_counts[module_key] += sum(param.numel() for param in params)

        if not param_groups:
            return None

        if "lora" in module_lrs and param_counts['lora'] == 0:
            logger.warning_rank0("lora learning_rate was set, but no trainable LoRA parameters were found.")
        if "audio_invert_tower" in module_lrs and param_counts['audio_invert_tower'] == 0:
            logger.warning_rank0(
                "audio_invert_tower learning_rate was set, but no trainable audio_invert_tower parameters were found."
            )
        if "listen_action" in module_lrs and param_counts['listen_action'] == 0:
            logger.warning_rank0(
                "listen_action learning_rate was set, but no trainable listen action parameters were found."
            )
        if "speak_action" in module_lrs and param_counts['speak_action'] == 0:
            logger.warning_rank0(
                "speak_action learning_rate was set, but no trainable speak_action_refinement_head parameters were found."
            )
        if "listen_semantic" in module_lrs and param_counts['listen_semantic'] == 0:
            logger.warning_rank0(
                "listen_semantic learning_rate was set, but no trainable listen semantic parameters were found."
            )
        if "speak_input_gate" in module_lrs and param_counts['speak_input_gate'] == 0:
            logger.warning_rank0(
                "speak_input_gate learning_rate was set, but speak_action_refinement_head.input_gate was not trainable."
            )

        optim_class, optim_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        optimizer = optim_class(param_groups, **optim_kwargs)
        lr_parts = [f"base={base_lr:.3e} ({param_counts['base']} params)"]
        if param_counts['lora'] > 0 or "lora" in module_lrs:
            lr_parts.append(f"lora={lora_lr:.3e} ({param_counts['lora']} params)")
        if param_counts['audio_invert_tower'] > 0 or "audio_invert_tower" in module_lrs:
            lr_parts.append(f"audio_invert_tower={audio_invert_lr:.3e} ({param_counts['audio_invert_tower']} params)")
        if param_counts['listen_action'] > 0 or "listen_action" in module_lrs:
            lr_parts.append(f"listen_action={listen_action_lr:.3e} ({param_counts['listen_action']} params)")
        if param_counts['speak_action'] > 0 or "speak_action" in module_lrs:
            lr_parts.append(f"speak_action={speak_action_lr:.3e} ({param_counts['speak_action']} params)")
        if param_counts['listen_semantic'] > 0 or "listen_semantic" in module_lrs:
            lr_parts.append(f"listen_semantic={listen_semantic_lr:.3e} ({param_counts['listen_semantic']} params)")
        if param_counts['speak_input_gate'] > 0 or "speak_input_gate" in module_lrs:
            lr_parts.append(f"speak_input_gate={speak_input_gate_lr:.3e} ({param_counts['speak_input_gate']} params)")
        logger.info_rank0("Using FunAudioChat module-wise learning rates: %s.", ", ".join(lr_parts))
        return optimizer

    def _log_funaudio_training_summary(self) -> None:
        if not self.is_world_process_zero():
            return

        train_samples = "unknown"
        eval_samples = "none"
        try:
            train_samples = f"{len(self.train_dataset):,}" if self.train_dataset is not None else "none"
        except TypeError:
            train_samples = "streaming"
        try:
            eval_samples = f"{len(self.eval_dataset):,}" if self.eval_dataset is not None else "none"
        except TypeError:
            eval_samples = "streaming"

        total_params = 0
        trainable_params = 0
        buckets: dict[str, int] = {}
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                continue
            count = param.numel()
            trainable_params += count
            bucket = _get_trainable_bucket(name)
            buckets[bucket] = buckets.get(bucket, 0) + count

        bucket_summary = ", ".join(
            f"{name}={_format_param_count(count)}" for name, count in sorted(buckets.items()) if count > 0
        ) or "none"
        logger.info_rank0(
            "FunAudioChat training summary: train_samples=%s, eval_samples=%s, "
            "trainable=%s/%s (%.4f%%), modules=[%s]",
            train_samples,
            eval_samples,
            _format_param_count(trainable_params),
            _format_param_count(total_params),
            100.0 * trainable_params / max(total_params, 1),
            bucket_summary,
        )

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
            if self.optimizer is None:
                self.optimizer = self._create_funaudio_module_lr_optimizer()
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _record_funaudio_losses(self, outputs: Any) -> None:
        if outputs is None:
            return
        for key in (
            "text_loss",
            "speech_loss",
            "listen_action_loss",
            "speak_action_loss",
        ):
            value = getattr(outputs, key, None)
            if value is None and isinstance(outputs, dict):
                value = outputs.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.numel() == 0:
                    continue
                value = value.detach().float().mean().item()
            try:
                self._funaudio_loss_buffer.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                continue

    def _flush_funaudio_loss_logs(self, logs: dict[str, float]) -> None:
        if not hasattr(self, "_funaudio_loss_buffer"):
            return
        if "loss" in logs:
            prefix = ""
        elif "eval_loss" in logs:
            prefix = "eval_"
        else:
            return
        for key, values in self._funaudio_loss_buffer.items():
            if not values:
                continue
            logs[f"{prefix}{key}"] = float(np.mean(values))
            values.clear()
        if hasattr(self, "_funaudio_grad_buffer") and prefix == "":
            for key, values in self._funaudio_grad_buffer.items():
                if not values:
                    continue
                logs[key] = float(np.mean(values))
                values.clear()

    @staticmethod
    def _get_funaudio_grad_bucket(name: str) -> str:
        name = name.removeprefix("module.")
        if name.startswith("listen_context_aggregator."):
            return "listen_context_grad_norm"
        if name.startswith("listen_action_refinement_head."):
            return "listen_action_grad_norm"
        if name.startswith("speak_action_refinement_head."):
            return "speak_action_grad_norm"
        if name.startswith("action_refinement_head."):
            return "action_grad_norm"
        if name.startswith("audio_invert_tower."):
            return "audio_invert_grad_norm"
        if "lora_" in name or ".lora_" in name:
            return "llm_lora_grad_norm"
        return "other_grad_norm"

    @staticmethod
    def _should_log_funaudio_grad_norms() -> bool:
        return os.environ.get("FUNAUDIOCHAT_LOG_GRAD_NORMS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _record_funaudio_grad_norms(self) -> None:
        if not self._should_log_funaudio_grad_norms():
            return
        if not hasattr(self, "_funaudio_grad_buffer"):
            return

        # Non-hook fallback for paths where gradients remain readable after backward.
        norm_keys = [key for key in self._funaudio_grad_buffer if key.endswith("_grad_norm")]
        sq_sums = {key: 0.0 for key in norm_keys}
        counts = {key: 0 for key in norm_keys}
        nonfinite_counts = {key.replace("_grad_norm", "_grad_nonfinite"): 0.0 for key in norm_keys}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.numel() == 0:
                continue
            bucket = self._get_funaudio_grad_bucket(name)
            grad_for_norm = grad.detach()
            finite_mask = torch.isfinite(grad_for_norm)
            nonfinite_counts[bucket.replace("_grad_norm", "_grad_nonfinite")] += float((~finite_mask).sum().item())
            grad_for_norm = torch.where(finite_mask, grad_for_norm, torch.zeros_like(grad_for_norm))
            sq_sums[bucket] += grad_for_norm.double().pow(2).sum().item()
            counts[bucket] += 1

        for key, sq_sum in sq_sums.items():
            self._funaudio_grad_buffer[key].append(math.sqrt(max(sq_sum, 0.0)) if counts[key] > 0 else 0.0)
        for key, count in nonfinite_counts.items():
            self._funaudio_grad_buffer[key].append(float(count))

    def _unwrap_funaudio_model_for_save(self) -> torch.nn.Module:
        try:
            return self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        except Exception:
            return self.model

    @staticmethod
    def _cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                result[key] = value.detach().cpu().contiguous()
        return result

    def _save_tensor_state(self, state_dict: dict[str, torch.Tensor], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        state_dict = self._cpu_state_dict(state_dict)
        if self.args.save_safetensors:
            safe_save_file(state_dict, output_path, metadata={"format": "pt"})
        else:
            torch.save(state_dict, output_path)

    @staticmethod
    def _strip_prefix_state(
        state_dict: dict[str, torch.Tensor],
        prefixes: tuple[str, ...],
        *,
        keep_prefix: bool = False,
    ) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    out_key = key if keep_prefix else key[len(prefix):]
                    if out_key.startswith("."):
                        out_key = out_key[1:]
                    result[out_key] = value
                    break
        return result

    @staticmethod
    def _filter_lora_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in state_dict.items():
            if not key.startswith("language_model."):
                continue
            if "lora_" not in key and "modules_to_save" not in key:
                continue
            result[key.removeprefix("language_model.")] = value
        return result

    @staticmethod
    def _filter_audio_invert_lora_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in state_dict.items():
            if not key.startswith("audio_invert_tower."):
                continue
            if "lora_" not in key and "modules_to_save" not in key:
                continue
            result[key.removeprefix("audio_invert_tower.")] = value
        return result

    @staticmethod
    def _has_funaudio_compact_tensors(state_dict: Optional[dict[str, torch.Tensor]]) -> bool:
        if not isinstance(state_dict, dict) or not state_dict:
            return False
        prefixes = (
            "language_model.",
            "listen_context_aggregator.",
            "listen_action_refinement_head.",
            "speak_action_refinement_head.",
            "audio_invert_tower.",
        )
        seen = {prefix: False for prefix in prefixes}
        for key in state_dict.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    seen[prefix] = True
        return all(seen.values())

    def _remove_stale_full_model_files(self, output_dir: str) -> None:
        stale_names = [
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ]
        root = os.path.abspath(output_dir)
        for name in stale_names:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                os.remove(path)
        for name in os.listdir(root):
            if name.startswith("model-") and name.endswith(".safetensors"):
                os.remove(os.path.join(root, name))
            elif name.startswith("pytorch_model-") and name.endswith(".bin"):
                os.remove(os.path.join(root, name))

    def _save_funaudio_manifest(
        self,
        *,
        output_dir: str,
        trained_modules_dir: str,
        module_counts: dict[str, int],
        module_files: dict[str, str],
    ) -> None:
        unwrapped = self._unwrap_funaudio_model_for_save()
        config = getattr(unwrapped, "config", None)
        config_overrides = {}
        if config is not None:
            for key in _FUN_AUDIO_CONFIG_KEYS:
                if hasattr(config, key):
                    value = getattr(config, key)
                    try:
                        json.dumps(value)
                        config_overrides[key] = value
                    except TypeError:
                        config_overrides[key] = str(value)

        yaml_config = _get_current_yaml_config() or {}
        flash_attn = str(yaml_config.get("flash_attn", "")).strip().lower()
        attn_implementation = None
        if flash_attn in {"fa2", "flash_attention_2", "flash-attn-2"}:
            attn_implementation = "flash_attention_2"
        elif flash_attn in {"eager", "sdpa"}:
            attn_implementation = flash_attn

        manifest = {
            "format": "funaudiochat_trained_modules_v1",
            "base_model": yaml_config.get("model_name_or_path"),
            "modules": module_files,
            "config_overrides": config_overrides,
        }
        if attn_implementation is not None:
            manifest["attn_implementation"] = attn_implementation
        manifest_path = os.path.join(output_dir, trained_modules_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _remove_funaudio_auxiliary_files(output_dir: str) -> None:
        """Keep compact outputs limited to trained_modules/ only."""
        for name in (
            "README.md",
            "added_tokens.json",
            "all_results.json",
            "chat_template.jinja",
            "merges.txt",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "train_results.json",
            "trainer_log.jsonl",
            "trainer_state.json",
            "training_args.bin",
            "vocab.json",
        ):
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                os.remove(path)
        for name in os.listdir(output_dir):
            if name.startswith("training_") and name.endswith(".png"):
                path = os.path.join(output_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
        for name in ("speech_tokenizer",):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)

    @staticmethod
    def _remove_peft_auxiliary_files(module_dir: str) -> None:
        readme_path = os.path.join(module_dir, "README.md")
        if os.path.isfile(readme_path):
            os.remove(readme_path)

    def _save_funaudio_trained_modules_only(self, output_dir: str, state_dict=None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_config = _get_funaudio_save_config()
        trained_modules_dir = save_config["trained_modules_dir"]
        trained_root = os.path.join(output_dir, trained_modules_dir)
        if os.path.isdir(trained_root):
            shutil.rmtree(trained_root)
        os.makedirs(trained_root, exist_ok=True)
        logger.info_rank0(f"Saving FunAudioChat trained modules to {trained_root}")

        module_counts: dict[str, int] = {}
        module_files: dict[str, str] = {}
        weight_name = "model.safetensors" if self.args.save_safetensors else "pytorch_model.bin"

        unwrapped = self._unwrap_funaudio_model_for_save()
        # Prefer the state_dict prepared by Trainer/Accelerate when it contains
        # the full FunAudioChat module set. Under distributed training this is
        # the safest source for final synchronized weights. Some PEFT paths pass
        # a filtered dict, so keep the unwrapped model as a fallback.
        if self._has_funaudio_compact_tensors(state_dict):
            logger.info_rank0("Using Trainer-provided state_dict for FunAudioChat compact save.")
        else:
            if state_dict is not None:
                logger.warning_rank0(
                    "Trainer-provided state_dict is incomplete for FunAudioChat compact save; "
                    "falling back to unwrapped model state_dict."
                )
            state_dict = unwrapped.state_dict()
        language_model = getattr(unwrapped, "language_model", None)
        lora_state = self._filter_lora_state(state_dict)
        if language_model is not None and hasattr(language_model, "save_pretrained") and lora_state:
            lora_dir = os.path.join(trained_root, "lora")
            os.makedirs(lora_dir, exist_ok=True)
            language_model.save_pretrained(
                lora_dir,
                state_dict=self._cpu_state_dict(lora_state),
                safe_serialization=self.args.save_safetensors,
            )
            self._remove_peft_auxiliary_files(lora_dir)
            module_counts["lora"] = len(lora_state)
            module_files["lora"] = os.path.join(trained_modules_dir, "lora")
        elif lora_state:
            lora_dir = os.path.join(trained_root, "lora")
            self._save_tensor_state(lora_state, os.path.join(lora_dir, weight_name))
            module_counts["lora"] = len(lora_state)
            module_files["lora"] = os.path.join(trained_modules_dir, "lora", weight_name)
        else:
            logger.warning_rank0("No LoRA tensors found while saving FunAudioChat trained modules.")

        audio_invert_tower = getattr(unwrapped, "audio_invert_tower", None)
        audio_lora_state = self._filter_audio_invert_lora_state(state_dict)
        if audio_invert_tower is not None and hasattr(audio_invert_tower, "save_pretrained") and audio_lora_state:
            audio_lora_dir = os.path.join(trained_root, "audio_invert_tower_lora")
            os.makedirs(audio_lora_dir, exist_ok=True)
            audio_invert_tower.save_pretrained(
                audio_lora_dir,
                state_dict=self._cpu_state_dict(audio_lora_state),
                safe_serialization=self.args.save_safetensors,
            )
            self._remove_peft_auxiliary_files(audio_lora_dir)
            module_counts["audio_invert_tower_lora"] = len(audio_lora_state)
            module_files["audio_invert_tower_lora"] = os.path.join(trained_modules_dir, "audio_invert_tower_lora")
        elif audio_lora_state:
            audio_lora_dir = os.path.join(trained_root, "audio_invert_tower_lora")
            self._save_tensor_state(audio_lora_state, os.path.join(audio_lora_dir, weight_name))
            module_counts["audio_invert_tower_lora"] = len(audio_lora_state)
            module_files["audio_invert_tower_lora"] = os.path.join(trained_modules_dir, "audio_invert_tower_lora", weight_name)

        module_specs = {
            "listen_context_aggregator": (("listen_context_aggregator.",), False),
            "listen_action_refinement_head": (("listen_action_refinement_head.",), False),
            "speak_action_refinement_head": (("speak_action_refinement_head.",), False),
        }
        if not audio_lora_state:
            module_specs["audio_invert_tower"] = (("audio_invert_tower.",), False)
        for module_name, (prefixes, keep_prefix) in module_specs.items():
            module_state = self._strip_prefix_state(state_dict, prefixes, keep_prefix=keep_prefix)
            if not module_state:
                logger.warning_rank0(f"No tensors found for {module_name} while saving FunAudioChat trained modules.")
                continue
            module_dir = os.path.join(trained_root, module_name)
            self._save_tensor_state(module_state, os.path.join(module_dir, weight_name))
            module_counts[module_name] = len(module_state)
            module_files[module_name] = os.path.join(trained_modules_dir, module_name, weight_name)

        self._save_funaudio_manifest(
            output_dir=output_dir,
            trained_modules_dir=trained_modules_dir,
            module_counts=module_counts,
            module_files=module_files,
        )
        self._remove_stale_full_model_files(output_dir)
        self._remove_funaudio_auxiliary_files(output_dir)

    @override
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        save_config = _get_funaudio_save_config()
        if save_config["save_trained_modules_only"]:
            return self._save_funaudio_trained_modules_only(output_dir, state_dict=state_dict)
        return super()._save(output_dir=output_dir, state_dict=state_dict)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return_outputs = bool(kwargs.pop("return_outputs", False))
        outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")
        if loss is None:
            raise ValueError("FunAudioChat model returned no loss from the training batch.")
        self._record_funaudio_losses(outputs)
        return (loss, outputs) if return_outputs else loss

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        self._record_funaudio_grad_norms()
        return loss

    @staticmethod
    def _round_significant(value: float, digits: int = 5) -> float:
        if value == 0.0 or not math.isfinite(value):
            return value
        return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)

    @classmethod
    def _round_log_values(cls, logs: dict[str, float], digits: int = 4) -> None:
        for key, value in list(logs.items()):
            if not isinstance(value, float):
                continue
            if key in {"learning_rate", "lr"}:
                logs[key] = cls._round_significant(value, digits=5)
            else:
                logs[key] = round(value, digits)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        self._flush_funaudio_loss_logs(logs)
        self._round_log_values(logs)
        return super().log(logs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
