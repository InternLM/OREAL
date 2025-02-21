# Copyright (c) InternLM. All rights reserved.
import json
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timedelta

import fire
import torch
import torch.distributed as dist
from mmengine import mkdir_or_exist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import is_flash_attn_2_available
from xtuner._lite import get_device, get_logger, get_torch_device_module
from xtuner._lite.accelerate import profile_time_and_memory, unpack_sequence
from xtuner._lite.algorithms.sft import SftCollator
from xtuner._lite.modelings import register_remote_code
from xtuner._lite.parallel import (
    ParallelSampler,
    setup_parallel,
    split_for_sequence_parallel,
)
from xtuner._lite.patches import AutoPatch, FSDPConfig
from xtuner._lite.patches.utils import pad_to_max_length, pad_to_multiple_of

from oreal.datasets import (
    InferDataset,
    OrealPromptDataset,
    PromptCollator,
    TrajectoryCollator,
    TrajectoryDataset,
    TrajectoryDatasetWithFilter,
)
from oreal.judgers import ParallelRouter
from oreal.utils import Config

logger = get_logger()

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


torch._dynamo.config.cache_size_limit = 16384


class RLParallelSampler(ParallelSampler):
    def __iter__(self):
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[
                : self.total_size
            ]

        # subsample
        chunk_size = len(indices) // self.world_size
        start = self.rank * chunk_size
        end = start + chunk_size
        indices = indices[start:end]

        return iter(indices[self.step :])


def log_format(rank, debug=False):

    formatter = f"[XTuner][RANK {rank}]"
    formatter += "[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]"

    if debug:
        formatter += "[<cyan>{name}</cyan>:"
        formatter += "<cyan>{function}</cyan>:"
        formatter += "<cyan>{line}</cyan>]"

    formatter += " <level>{message}</level>"
    return formatter


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


def reduce_mean(data, group):
    data_tensor = torch.tensor(data, device=DEVICE)
    dist.all_reduce(data_tensor, op=dist.ReduceOp.AVG, group=group)
    return data_tensor.item()


def threshold_rescale(prob, threshold=0.5):
    prob = prob - threshold
    prob = prob / (1 - threshold)
    prob = prob.clamp(0, 1)
    return prob


def topk_rescale(prob, topk_ratio=0.5):
    topk_num = int(prob.numel() * topk_ratio)
    values, indices = torch.topk(prob, topk_num)
    result = torch.zeros_like(prob)
    if values.max() != values.min():
        normalized_values = (values - values.min()) / (values.max() - values.min())
    else:
        normalized_values = torch.ones_like(values)
    result[indices] = normalized_values
    return result


def train_oreal(cfg_path, **kwargs):
    args = Config.fromfile(cfg_path)
    args.update(kwargs)

    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    register_remote_code()

    setup_parallel()
    set_random_seed(args.seed)

    rank = dist.get_rank()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    args.work_dir = os.path.join(args.work_dir, timestamp)
    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f"rank{rank}.log")

    # Change the log format printed in the terminal
    lvl = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=lvl, format=log_format(rank, args.debug))
    # Change the format saved in the log file
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

    logger.info(args)
    if rank == 0:
        env = collect_env()
        import transformers
        import xtuner

        env["Transformers"] = transformers.__version__
        env["XTuner"] = f"{xtuner.__version__}+{get_git_hash(digits=6)}"
        runtime_env = OrderedDict()
        runtime_env.update(env)
        runtime_env["Seed"] = args.seed
        runtime_env["World Size"] = dist.get_world_size()

        runtime_env_info = "\n    " + "\n    ".join(f"{k}: {v}" for k, v in runtime_env.items())
        dash_line = "-" * 60
        logger.info("\n" + dash_line + "\nRuntime environment:" + runtime_env_info + "\n" + dash_line + "\n")
    # -------------------    Environment  End  ------------------------------ #

    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################
    if args.dtype == "auto":
        args.dtype = "bf16" if DEVICE_MODULE.is_bf16_supported() else "fp16"

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        if DEVICE_MODULE.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            raise RuntimeError("The device does not support `bf16`, " "please set `dtype` to `fp16`.")
    else:
        raise RuntimeError("`dtype` only supports `fp16`, `bf16` or `auto`, " f"but found {args.dtype}.")

    with torch.device("meta"):
        # In order to save CPU memory and GPU memory,
        # initialize an empty complete model on all ranks first.
        # At the same time, a non-empty complete model will be loaded
        # on the CPU of rank0.
        # After the model is parallelized, the parameters of the complete
        # model on rank0 will be loaded.
        actor_model = AutoModelForCausalLM.from_pretrained(args.actor, attn_implementation="flash_attention_2", torch_dtype=dtype)

        for module in actor_model.modules():
            for p_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                    setattr(module, p_name, param_fp32)

        ref_model = AutoModelForCausalLM.from_pretrained(args.reference, attn_implementation="flash_attention_2", torch_dtype=dtype)

        for param in ref_model.parameters():
            param.requires_grad = False

        if args.token_level_rm is not None:
            token_level_rm = AutoModelForCausalLM.from_pretrained(
                args.token_level_rm, attn_implementation="flash_attention_2", torch_dtype=dtype
            )
            # replace the language model head with a reward model linear head
            token_level_rm.lm_head = torch.nn.Linear(token_level_rm.config.hidden_size, 1, bias=False)

            for module in token_level_rm.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        # Ensure all numerical values in the optimizer are fp32.
                        # Don't worry about speed, FSDP will use `dtype`
                        # during forward.
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)

    with profile_time_and_memory("[Parallelize Actor]"):
        actor_model = AutoPatch.from_causal_lm(
            actor_model,
            fsdp_config=FSDPConfig(
                tp_size=args.tp_size,
                sp_size=args.sp_size,
                param_dtype=dtype,
                reduce_dtype=dtype,
                cpu_offload=args.cpu_offload,
                reshard_after_forward=False,
                mesh_prefix="actor",
            ),
        )
    dist.barrier()

    with profile_time_and_memory("[Parallelize Reference]"):
        ref_model = AutoPatch.from_causal_lm(
            ref_model,
            fsdp_config=FSDPConfig(
                tp_size=args.tp_size,
                sp_size=args.sp_size,
                param_dtype=dtype,
                reduce_dtype=dtype,
                cpu_offload=args.cpu_offload,
                reshard_after_forward=True,
                mesh_prefix="ref",
            ),
        )
    dist.barrier()

    if args.token_level_rm is not None:
        with profile_time_and_memory("[Parallelize Reward]"):
            token_level_rm = AutoPatch.from_causal_lm(
                token_level_rm,
                fsdp_config=FSDPConfig(
                    tp_size=args.tp_size,
                    sp_size=args.sp_size,
                    param_dtype=dtype,
                    reduce_dtype=dtype,
                    cpu_offload=args.cpu_offload,
                    reshard_after_forward=True,
                    mesh_prefix="reward",
                ),
            )
            token_level_rm.train()
            # print head weight
            # logger.info(f"Rank {rank} Reward model head weight: {token_level_rm.patched_model.lm_head.weight}")
    dist.barrier()
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################
    actor_sp_mesh = actor_model.sequence_parallel_mesh
    actor_dp_mesh = actor_model.data_parallel_mesh
    actor_data_mesh = actor_model.data_mesh
    actor_dp_size = actor_dp_mesh.size()

    actor_sp_size = actor_sp_mesh.size()

    prompt_global_batch = args.gen_global_batch // args.prompt_repeat_k

    tokenizer = AutoTokenizer.from_pretrained(args.actor, trust_remote_code=True, padding_side="right")

    if args.chat_template is not None:
        if rank == 0:
            logger.info(f"[CHAT_TEMPLATE] {args.chat_template}")
        tokenizer.chat_template = args.chat_template

    stop_token_ids = []
    word_ids = tokenizer.encode(args.stop_word, add_special_tokens=False)
    if len(word_ids) > 1:
        raise NotImplementedError("The stop word must be a single token.")
    stop_token_ids.append(word_ids[0])

    with profile_time_and_memory("[Dataset & Dataloader]"):

        prompt_dataset = OrealPromptDataset(
            args.datasets,
            tokenizer,
            difficulty_balance_cfg=args.data_difficulty_balance_cfg,
        )
        if rank == 0:
            logger.info(f"[Dataset] {len(prompt_dataset)} prompts.")

        assert is_flash_attn_2_available()
        prompt_collator = PromptCollator(pack_batch=True)
        prompt_sampler = ParallelSampler(prompt_dataset, actor_dp_mesh, prompt_global_batch, shuffle=True)

        prompt_dataloader = DataLoader(
            prompt_dataset,
            batch_size=prompt_global_batch // actor_dp_mesh.size(),
            num_workers=args.num_workers,
            # Ensure to round up or drop last based on the `global_batch_size`,
            # if you want to replace a custom sampler.
            sampler=prompt_sampler,
            collate_fn=prompt_collator,
            persistent_workers=args.num_workers > 0,
        )

        if rank == 0:
            logger.info(f"[Dataloader] {len(prompt_dataloader)} batches.")
            _first_batch = [prompt_dataset[i] for i in range(prompt_global_batch)]
            logger.debug(f"[Dataloader] Training Batch:\n{_first_batch}")

    dist.barrier()
    # -------------------    Dataset & Dataloader  End  --------------------- #

    # ---------------------    Router  Start  ------------------------------- #
    judger_router = ParallelRouter(
        judgers_config=args.judgers_config,
        data_judger_mapping=args.data_judger_mapping,
        logger=logger,
    )

    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    actor_params = [p for p in actor_model.parameters() if p.requires_grad]
    actor_optimizer = AdamW(actor_params, lr=args.actor_lr, weight_decay=args.wd)

    if args.token_level_rm is not None:
        token_rm_params = [p for p in token_level_rm.parameters() if p.requires_grad]
        token_rm_optimizer = AdamW(token_rm_params, lr=args.token_level_rm_lr, weight_decay=args.wd)

    total_steps = args.total_steps
    if total_steps > len(prompt_dataloader):
        logger.warning(f"Total steps {total_steps} is greater than the number of prompts {len(prompt_dataloader)}, set to dataloader length.")
        total_steps = len(prompt_dataloader)

    warmup_steps = args.warmup_steps
    rm_warmup_steps = args.get("rm_warmup_steps", warmup_steps)
    lr_min = args.get("actor_min_lr", args.actor_lr)
    token_level_rm_lr_min = args.get("token_level_rm_lr_min", args.token_level_rm_lr)

    if args.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(actor_optimizer, warmup_fn)
    cosine_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_steps - warmup_steps, eta_min=lr_min)

    if args.token_level_rm is not None:

        def rm_warmup_fn(x):
            return x / rm_warmup_steps if x < rm_warmup_steps else 1

        token_rm_warmup_scheduler = LambdaLR(token_rm_optimizer, rm_warmup_fn)
        token_rm_cosine_scheduler = CosineAnnealingLR(token_rm_optimizer, T_max=total_steps - rm_warmup_steps, eta_min=token_level_rm_lr_min)

    # ----------------    Optimizer & Scheduler End   ----------------------- #

    ###########################################################################
    #                          5. Training                                    #
    ###########################################################################

    if args.filter_trajectory:
        trajectory_dataset = TrajectoryDatasetWithFilter(repeat_k=args.prompt_repeat_k)
    else:
        trajectory_dataset = TrajectoryDataset()

    prompt_iterator = iter(prompt_dataloader)

    start_step = 0
    start_train_t = time.time()
    DEVICE_MODULE.empty_cache()
    DEVICE_MODULE.reset_peak_memory_stats()
    max_memory = DEVICE_MODULE.max_memory_allocated()
    logger.info("[Train] Begin Train Loop. The current GPU memory is " f"{(max_memory / 1024**3):.1f}GB")

    for step in range(start_step, total_steps):

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_last_lr()[0]
            if args.token_level_rm is not None:
                token_rm_warmup_scheduler.step()
                token_rm_cur_lr = token_rm_warmup_scheduler.get_last_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_last_lr()[0]
            if args.token_level_rm is not None:
                token_rm_cosine_scheduler.step()
                token_rm_cur_lr = token_rm_cosine_scheduler.get_last_lr()[0]

        DEVICE_MODULE.reset_peak_memory_stats()

        step_kl_penalty_loss = 0
        step_rl_loss = 0
        step_token_level_rm_loss = 0
        step_start_t = time.time()
        step_positive_loss = 0
        step_negative_loss = 0

        if step < args.actor_freeze_steps:
            # Only update the parameters of the token-level reward model
            update_actor = False
        else:
            update_actor = True

        DEVICE_MODULE.reset_peak_memory_stats()

        data = next(prompt_iterator)
        prompt_input_ids = unpack_sequence(data["input_ids"].to(DEVICE), data["num_tokens"])
        infer_num_tokens = data["num_tokens"].to(DEVICE)
        # repeat prompt for k times
        prompt_input_ids = [p for p in prompt_input_ids for _ in range(args.prompt_repeat_k)]  # AAAABBBBCCCC
        infer_num_tokens = torch.Tensor([n for n in infer_num_tokens for _ in range(args.prompt_repeat_k)])
        message_data = [m for m in data["message_data"] for _ in range(args.prompt_repeat_k)]
        metadata = [m for m in data["metadata"] for _ in range(args.prompt_repeat_k)]

        # Stage 1,  Actor Model Generation
        step_avg_new_tokens = 0
        step_gen_start_t = time.time()

        actor_model.eval()
        # During the generation stage, sequence parallelism was not used,
        # even when the sp size is greater than 1.
        # Per sp rank processes different prompts in parallel.
        responses = actor_model.generate(
            prompt_input_ids,
            stop_token_ids,
            max_length=args.gen_max_length,
            max_batch_size=len(prompt_input_ids),
            max_prefill_batch=args.max_prefill_batch,
            max_new_tokens=args.gen_max_new,
            do_sample=args.gen_do_sample,
            top_k=args.gen_top_k,
            top_p=args.gen_top_p,
            temperature=args.temperature,
            cuda_graph=args.cuda_graph,
        )

        # decode responses
        response_texts = [tokenizer.decode(res, skip_special_tokens=False) for res in responses]

        actor_model.train()
        dist.barrier()

        step_avg_new_tokens = sum([len(res) for res in responses]) / len(responses)
        step_gen_time = time.time() - step_gen_start_t

        prompt_input_ids = [p[0].tolist() for p in prompt_input_ids]

        # Stage 2,  Infer
        step_infer_start_t = time.time()
        step_infer_consumed_tokens = 0

        # submit to judger
        if actor_data_mesh.get_local_rank() == 0:
            submit_batch = []
            for i in range(len(message_data)):
                submit_batch.append(
                    {
                        "prompt_messages": message_data[i],
                        "completion_messages": [{"role": "assistant", "content": response_texts[i]}],
                        "metadata": metadata[i],
                    }
                )
            token, indexes_for_local = judger_router.submit(submit_batch)

        # `infer_dataset` varies at each dp rank, there is no need to
        # use the parallel sampler.
        infer_dataset = InferDataset(
            prompt_input_ids,
            responses,
            message_data,
            metadata,
        )
        infer_dataloader = DataLoader(
            infer_dataset,
            batch_size=args.rl_mirco_batch,
            num_workers=0,
            collate_fn=SftCollator(pack_batch=True),
            shuffle=False,
            persistent_workers=False,
        )

        policies = []
        for infer_packed_seq in infer_dataloader:
            # labels are already shifted in InferDataset
            infer_labels = infer_packed_seq["labels"].to(DEVICE)
            infer_input_ids = infer_packed_seq["input_ids"].to(DEVICE)
            infer_num_tokens = infer_packed_seq["num_tokens"].to(DEVICE)
            infer_batch_size = infer_num_tokens.numel()

            step_infer_consumed_tokens += infer_num_tokens.sum() / actor_data_mesh.size()

            unpacked_input_ids = unpack_sequence(infer_input_ids, infer_num_tokens, dim=1)
            unpacked_labels = unpack_sequence(infer_labels, infer_num_tokens, dim=1)

            for i in range(infer_batch_size):
                assert unpacked_input_ids[i].numel() == infer_num_tokens[i]
                assert unpacked_labels[i].numel() == infer_num_tokens[i]

                _policy = {
                    "input_ids": unpacked_input_ids[i].flatten().tolist(),
                    "labels": unpacked_labels[i].flatten().tolist(),
                    "num_tokens": infer_num_tokens[i].item(),
                }
                _policy["sequence_text"] = tokenizer.decode(_policy["input_ids"], skip_special_tokens=False)
                policies.append(_policy)

        step_infer_time = time.time() - step_infer_start_t

        # ------------------------------------------------------------- #
        # --------------------------Get Judger Reward------------------ #
        # ------------------------------------------------------------- #
        # query results from judger
        if actor_data_mesh.get_local_rank() == 0:
            while True:
                try:
                    judger_results = judger_router.query(token, timeout=3)
                    logger.info(f"Query judger results: {judger_results}")
                    break
                except TimeoutError as e:
                    logger.info(f"Judger query timeout: {e}. Will retry")
            judger_rewards = [list(r.values())[0] for r in judger_results]
            judger_rewards = [r if r is not None else -1.0 for r in judger_rewards]
            judger_rewards = torch.tensor(judger_rewards, dtype=torch.float32).to(DEVICE)
        else:
            judger_rewards = torch.tensor([0] * len(policies), dtype=torch.float32).to(DEVICE)

        dist.barrier()
        # broadcast judger rewards to same data mesh
        dist.all_reduce(judger_rewards, op=dist.ReduceOp.SUM, group=actor_data_mesh.get_group())

        # reward shaping, use GRPO or RLOO to normalize rewards
        _rewards = judger_rewards.reshape(-1, args.prompt_repeat_k).T
        if args.reward_shaping_type == "rloo":
            baseline = (_rewards.sum(0) - _rewards) / (args.prompt_repeat_k - 1)
            judger_advantages = _rewards - baseline
        elif args.reward_shaping_type == "grpo":
            judger_advantages = (_rewards - _rewards.mean(0)) / (_rewards.std(0) + 1e-8)
        else:
            raise NotImplementedError(f"Reward shaping type {args.reward_shaping_type} is not implemented.")
        judger_advantages = judger_advantages.T.flatten()
        # update policies
        assert len(judger_rewards) == len(policies)
        for i in range(len(policies)):
            policies[i]["judger_reward"] = judger_rewards[i].item()
            policies[i]["judger_advantage"] = judger_advantages[i].item()

        # ------------------------------------------------------------- #
        # --------------------------Stage 4, RL------------------------ #
        # ------------------------------------------------------------- #
        # Stage 4, RL
        step_rl_start_t = time.time()

        _global_policies = [None] * actor_dp_size
        dist.all_gather_object(_global_policies, policies, actor_dp_mesh.get_group())

        global_policies = []
        for _rank_policies in _global_policies:
            global_policies.extend(_rank_policies)

        trajectory_dataset.update(global_policies)
        if rank == 0:
            # dump trajectory
            _buffer_dir = os.path.join(args.work_dir, "trajectories")
            mkdir_or_exist(_buffer_dir)
            _buffer_file = os.path.join(_buffer_dir, f"step.{step}.jsonl")
            trajectory_dataset.dump_jsonl(_buffer_file, tokenizer, args.debug)
            _buffer_log_file = os.path.join(_buffer_dir, f"step.{step}.log")
            trajectory_dataset.dump_log(_buffer_log_file, tokenizer, args.debug)

        rl_global_batch = args.rl_global_batch
        if args.filter_trajectory:
            _world_size = actor_dp_mesh.size()
            _data_size = len(trajectory_dataset)
            # train_global_batch is divisible by world_size
            rl_global_batch = _data_size // _world_size * _world_size

        rl_loader = DataLoader(
            trajectory_dataset,
            batch_size=args.rl_mirco_batch,
            num_workers=0,
            collate_fn=TrajectoryCollator(pack_batch=True),
            shuffle=False,
            sampler=RLParallelSampler(trajectory_dataset, actor_dp_mesh, rl_global_batch, shuffle=False),
            persistent_workers=False,
        )

        # Count the total number of tokens used for training RL on all ranks
        # It is necessary for `per-token` loss, otherwise the number of tokens
        # for each backward is unbalanced.
        global_action_tokens = trajectory_dataset.num_action_tokens
        global_positive_tokens = sum(
            [(torch.tensor(t["labels"]) >= 0).sum().item() for t in trajectory_dataset._trajectories if t["judger_reward"] > 0]
        )
        global_negative_tokens = global_action_tokens - global_positive_tokens
        global_num_seqs = len(trajectory_dataset._trajectories)

        step_avg_judger_reward = sum([t["judger_reward"] for t in global_policies]) / len(global_policies)
        step_sum_gen_entropy = 0
        step_sum_ref_kl = 0
        step_action_tokens = 0
        step_rl_consumed_tokens = 0

        step_sum_adv = 0

        for packed_policy in rl_loader:

            rl_input_ids = packed_policy["input_ids"].to(DEVICE)
            rl_num_tokens = packed_policy["num_tokens"].to(DEVICE)
            assert rl_input_ids.numel() == rl_num_tokens.sum()
            rl_batch_size = rl_num_tokens.numel()
            # labels are already shifted in InferDataset
            rl_labels = packed_policy["labels"].to(DEVICE)

            judger_rewards = torch.Tensor(packed_policy["judger_rewards"]).to(DEVICE)  # shape: (rl_mirco_batch, )
            judger_advantages = torch.Tensor(packed_policy["judger_advantages"]).to(DEVICE)  # shape: (rl_mirco_batch, )

            actor_input_ids = rl_input_ids.clone()
            actor_labels = rl_labels.clone()
            actor_num_tokens = rl_num_tokens.clone().tolist()

            actor_cu_seq_lens = torch.cumsum(torch.IntTensor([0] + actor_num_tokens), dim=0).to(DEVICE).int()
            actor_position_ids = [torch.arange(num) for num in actor_num_tokens]
            actor_position_ids = torch.cat(actor_position_ids, dim=0).to(DEVICE).unsqueeze_(0)

            with nullcontext() if update_actor else torch.no_grad():
                packed_actor_logits = actor_model(
                    input_ids=actor_input_ids,
                    position_ids=actor_position_ids,
                    use_cache=False,
                    cu_seq_lens_q=actor_cu_seq_lens,
                    cu_seq_lens_k=actor_cu_seq_lens,
                    max_length_q=max(actor_num_tokens),
                    max_length_k=max(actor_num_tokens),
                    sequence_parallel_mesh=actor_sp_mesh,
                ).logits

            # -------sft loss--------
            # calculate sft loss on each sp(tp) rank and then gather them to dp rank, avoid gather logits which may lead to OOM
            if actor_model.fsdp_config.torch_compile:
                _actor_labels = pad_to_max_length(actor_labels, -100, actor_model.fsdp_config.max_length, 1)
            else:
                if actor_sp_mesh and actor_sp_mesh.size() > 1:
                    multiple_of = actor_sp_mesh.size() * actor_model.tp_mesh.size()
                else:
                    multiple_of = actor_model.tp_mesh.size()
                _actor_labels = pad_to_multiple_of(actor_labels, -100, multiple_of, 1)

            if actor_sp_mesh and actor_sp_mesh.size() > 1:
                _actor_labels = split_for_sequence_parallel(_actor_labels, dim=1, sp_mesh=actor_sp_mesh)

            if actor_model.tp_mesh.size() > 1:
                _actor_labels = split_for_sequence_parallel(_actor_labels, dim=1, sp_mesh=actor_model.tp_mesh)
            packed_sft_loss = F.cross_entropy(packed_actor_logits.squeeze(), _actor_labels.squeeze(), reduction="none").unsqueeze(
                0
            )  # shape: 1, seqlen

            if actor_model.tp_mesh.size() > 1:
                _packed_sft_loss = dist.nn.all_gather(packed_sft_loss, group=actor_model.tp_mesh.get_group())
                packed_sft_loss = torch.cat(_packed_sft_loss, dim=1)

            if actor_sp_mesh and actor_sp_mesh.size() > 1:
                _packed_sft_loss = dist.nn.all_gather(packed_sft_loss, group=actor_sp_mesh.get_group())
                packed_sft_loss = torch.cat(_packed_sft_loss, dim=1)

            packed_sft_loss = packed_sft_loss[:, : actor_labels.size(1)]

            # The labels of prefill tokens and last token are -100.
            # HACK: (for sp) The -100 part takes the value of 0,
            # this part will be masked later.
            packed_logprobs = actor_model.gather_logprobs(packed_actor_logits, actor_labels.clip(0), actor_sp_mesh)

            logprobs = unpack_sequence(packed_logprobs, actor_num_tokens, dim=1)
            sft_loss = unpack_sequence(packed_sft_loss, actor_num_tokens, dim=1)

            ref_input_ids = rl_input_ids.clone()
            ref_labels = rl_labels.clone()
            ref_num_tokens = rl_num_tokens.clone().tolist()

            ref_cu_seq_lens = torch.cumsum(torch.IntTensor([0] + ref_num_tokens), dim=0).to(DEVICE).int()
            ref_position_ids = [torch.arange(num) for num in ref_num_tokens]
            ref_position_ids = torch.cat(ref_position_ids, dim=0).to(DEVICE).unsqueeze_(0)

            with torch.no_grad():
                packed_ref_logits = ref_model(
                    input_ids=ref_input_ids,
                    position_ids=ref_position_ids,
                    use_cache=False,
                    cu_seq_lens_q=ref_cu_seq_lens,
                    cu_seq_lens_k=ref_cu_seq_lens,
                    max_length_q=max(ref_num_tokens),
                    max_length_k=max(ref_num_tokens),
                    sequence_parallel_mesh=actor_sp_mesh,
                ).logits

            if args.token_level_rm is not None:
                # assert ref_num_tokens.sum() == ref_input_ids.numel() * 8, f"{ref_num_tokens}, {_num_pad}, {ref_input_ids.numel()}, {rl_input_ids.numel()}"
                packed_rm_logits = token_level_rm(
                    input_ids=ref_input_ids,
                    position_ids=ref_position_ids,
                    use_cache=False,
                    cu_seq_lens_q=ref_cu_seq_lens,
                    cu_seq_lens_k=ref_cu_seq_lens,
                    max_length_q=max(ref_num_tokens),
                    max_length_k=max(ref_num_tokens),
                    sequence_parallel_mesh=actor_sp_mesh,
                ).logits
                # use last token logits as reward logits
                packed_rm_logits = packed_rm_logits[:, :, 0]  # TODO: replace with auto path rm
                if token_level_rm.tp_mesh.size() > 1:
                    _packed_rm_logits = dist.nn.all_gather(packed_rm_logits, group=token_level_rm.tp_mesh.get_group())
                    packed_rm_logits = torch.cat(_packed_rm_logits, dim=1)
                if actor_sp_mesh and actor_sp_mesh.size() > 1:
                    _packed_rm_logits = dist.nn.all_gather(packed_rm_logits, group=actor_sp_mesh.get_group())
                    packed_rm_logits = torch.cat(_packed_rm_logits, dim=1)
                packed_rm_logits = packed_rm_logits[:, : actor_labels.size(1)]
                rm_logits = unpack_sequence(packed_rm_logits, ref_num_tokens, dim=1)

            # The labels of prefill tokens and last token are -100.
            # HACK: (for sp) The -100 part takes the value of 0,
            # this part will be masked later.
            packed_ref_logprobs = ref_model.gather_logprobs(packed_ref_logits, ref_labels.clip(0), actor_sp_mesh)
            ref_logprobs = unpack_sequence(packed_ref_logprobs, ref_num_tokens, dim=1)
            unpacked_labels = unpack_sequence(rl_labels, rl_num_tokens, dim=1)

            _positive_losses = []
            _negative_losses = []
            _kl_penalty_losses = []
            _token_level_rm_losses = []
            _losses = []
            for i in range(rl_batch_size):
                _judger_reward = judger_rewards[i]
                assert unpacked_labels[i].numel() == rl_num_tokens[i]
                # from the last prefill token, to the second-to-last token (excluding the eos token)
                _num_action_tokens = (unpacked_labels[i] >= 0).sum()

                _logprobs = logprobs[i][0, -_num_action_tokens - 1 : -1]
                _ref_logprobs = ref_logprobs[i][0, -_num_action_tokens - 1 : -1]

                _old_logprobs = _logprobs.detach()
                _judger_advantages = judger_advantages[i]

                if args.token_level_rm is not None:
                    # compute cumulative mean of rm scores
                    _rm_scores = rm_logits[i][0, -_num_action_tokens - 1 : -1]
                    _cum_mean_rm_scores = _rm_scores.cumsum(0).squeeze() / torch.arange(1, _num_action_tokens + 1).to(DEVICE)
                    _seq_mean_rm_scores = _rm_scores.mean()

                    # ----------token level rm loss (cross entropy)------------
                    _rm_label = torch.tensor([int(max(_judger_reward, 0))]).to(DEVICE)
                    _seq_mean_rm_scores = _seq_mean_rm_scores.reshape(_rm_label.shape)
                    _token_level_rm_loss = F.binary_cross_entropy_with_logits(_seq_mean_rm_scores.float(), _rm_label.float(), reduction="none")
                    _token_level_rm_loss = _token_level_rm_loss.sum() * actor_dp_size / global_num_seqs
                    _token_level_rm_losses.append(_token_level_rm_loss)

                    # use probability to reweight policy loss
                    _correct_prob = torch.sigmoid(_cum_mean_rm_scores).detach()
                    _incorrect_prob = 1 - _correct_prob

                    if args.get("threshold_rescale", False):
                        correct_threshold = args.get("correct_threshold", 0.5)
                        incorrect_threshold = args.get("incorrect_threshold", 0.5)
                        _pos_weight = threshold_rescale(_correct_prob, correct_threshold)
                        _neg_weight = threshold_rescale(_incorrect_prob, incorrect_threshold)
                    elif args.get("topk_rescale", False):
                        correct_topk_ratio = args.get("correct_topk_ratio", 0.5)
                        incorrect_topk_ratio = args.get("incorrect_topk_ratio", 0.5)
                        _pos_weight = topk_rescale(_correct_prob, correct_topk_ratio)
                        _neg_weight = topk_rescale(_incorrect_prob, incorrect_topk_ratio)
                    else:
                        raise NotImplementedError("Only support threshold_rescale and topk_rescale.")
                else:
                    _pos_weight, _neg_weight = 1.0, 1.0

                # ----------positive loss (behavior cloning)------------
                _positive_loss = sft_loss[i][0, -_num_action_tokens - 1 : -1]
                _positive_loss = (_positive_loss * _pos_weight).sum()
                if args.get("pos_mult_adv", False):
                    _positive_loss = _positive_loss * _judger_advantages
                if _judger_reward > 0:
                    _positive_loss = _positive_loss * actor_dp_size / global_positive_tokens * args.positive_loss_factor
                else:
                    # negative sample does not need sft loss
                    _positive_loss = torch.zeros_like(_positive_loss)
                _positive_losses.append(_positive_loss)

                # ----------negative loss (policy gradient)------------
                if _judger_reward > 0:
                    # positive sample, does not need policy loss
                    _negative_loss = torch.zeros_like(_positive_loss)
                    _kl_penalty_loss = torch.zeros_like(_positive_loss)
                    _negative_losses.append(_negative_loss)
                else:
                    _advantages = _judger_advantages * _neg_weight
                    _negative_loss = torch.exp(_logprobs - _old_logprobs.detach()) * _advantages
                    _negative_loss = -torch.sum(_negative_loss) * actor_dp_size / global_negative_tokens * args.negative_loss_factor
                    _negative_losses.append(_negative_loss)

                # ----------compute kl penalty------------
                assert _logprobs.ndim == 1
                kl_type = args.get("kl_type", "unbias")  # kl, unbias, mse
                if kl_type == "kl":
                    kl = _ref_logprobs - _logprobs
                    _kl_penalty_loss = (args.kl_coef * kl).sum() * actor_dp_size / global_negative_tokens
                elif kl_type == "unbias":
                    kl = _ref_logprobs - _logprobs
                    nonneg_nobias_kl = torch.exp(kl) - kl - 1
                    _kl_penalty_loss = (args.kl_coef * nonneg_nobias_kl).sum() * actor_dp_size / global_negative_tokens
                elif kl_type == "mse":
                    _kl_penalty_loss = (
                        (args.kl_coef * (_ref_logprobs - _logprobs).square() / 2).sum() * actor_dp_size / global_negative_tokens
                    )
                _kl_penalty_losses.append(_kl_penalty_loss)

                # ----------compute total loss------------
                _loss = _positive_loss + _negative_loss + _kl_penalty_loss
                _losses.append(_loss)

                step_sum_gen_entropy += -_old_logprobs.sum().item()
                step_sum_ref_kl += (_old_logprobs - _ref_logprobs).sum().item()
                step_sum_adv += _judger_advantages.sum().item()
                step_action_tokens += _num_action_tokens.item()

            loss = sum(_losses)
            if update_actor:
                loss.backward()

            # for logging
            step_positive_loss += sum(_positive_losses).item()
            step_negative_loss += sum(_negative_losses).item()
            step_kl_penalty_loss += sum(_kl_penalty_losses).item()
            step_rl_loss += loss.item()
            step_rl_consumed_tokens += rl_num_tokens.sum() / actor_data_mesh.size()

            if args.token_level_rm is not None:
                token_level_rm_loss = sum(_token_level_rm_losses)
                token_level_rm_loss.backward()
                step_token_level_rm_loss += token_level_rm_loss.item()

        step_rl_time = time.time() - step_rl_start_t
        step_avg_ref_kl = step_sum_ref_kl / step_action_tokens
        step_avg_gen_entropy = step_sum_gen_entropy / step_action_tokens
        step_avg_adv = step_sum_adv / step_action_tokens

        actor_data_group = actor_data_mesh.get_group()
        step_avg_ref_kl = reduce_mean(step_avg_ref_kl, actor_data_group)
        step_avg_gen_entropy = reduce_mean(step_avg_gen_entropy, actor_data_group)
        step_avg_adv = reduce_mean(step_avg_adv, actor_data_group)
        step_avg_new_tokens = reduce_mean(step_avg_new_tokens, actor_data_group)

        if update_actor:
            actor_grad_norm = actor_model.clip_grad_norm(args.max_grad_norm)
            actor_grad_norm = actor_grad_norm.item()
            actor_optimizer.step()
            actor_optimizer.zero_grad()
        else:
            actor_grad_norm = 0

        if args.token_level_rm is not None:
            token_rm_grad_norm = token_level_rm.clip_grad_norm(args.max_grad_norm)
            token_rm_grad_norm = token_rm_grad_norm.item()
            token_rm_optimizer.step()
            token_rm_optimizer.zero_grad()

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))

        infer_tgs = int(step_infer_consumed_tokens / step_infer_time)
        rl_tgs = int(step_rl_consumed_tokens / step_rl_time)

        actor_lr = cur_lr if update_actor else 0.0
        max_memory = DEVICE_MODULE.max_memory_allocated()
        log_dict = {
            "step": step + 1,
            "actor_lr": actor_lr,
            "actor_grad_norm": actor_grad_norm,
            "token_level_rm_lr": token_rm_cur_lr if args.token_level_rm is not None else 0.0,
            "token_rm_grad_norm": token_rm_grad_norm if args.token_level_rm is not None else 0.0,
            "avg_judger_reward": step_avg_judger_reward,
            "avg_adv": step_avg_adv,
            "avg_gen_entropy": step_avg_gen_entropy,
            "avg_ref_kl": step_avg_ref_kl,
            "positive_loss": step_positive_loss,
            "negative_loss": step_negative_loss,
            "kl_penalty_loss": step_kl_penalty_loss,
            "rl_loss": step_rl_loss,
            "token_level_rm_loss": step_token_level_rm_loss if args.token_level_rm is not None else 0.0,
            "max_memory": max_memory / 1024**3,
            "avg_new_tokens": step_avg_new_tokens,
            "num_rl_tokens": step_rl_consumed_tokens,
            "infer_tgs": infer_tgs,
            "rl_tgs": rl_tgs,
            "gen_time": step_gen_time,
            "infer_time": step_infer_time,
            "rl_time": step_rl_time,
            "total_time": step_time,
            "eta": eta.seconds,
        }
        for key, value in log_dict.items():
            if isinstance(value, torch.Tensor):
                log_dict[key] = value.item()
        with open(os.path.join(args.work_dir, f"rank{rank}.log.jsonl"), "a") as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")

        if is_interval(step, total_steps, args.log_interval):
            logger.info(
                "[Train] Step "
                f"{step + 1}/{total_steps}  "
                f"actor_lr: {cur_lr:.3e}  "
                f"actor_grad_norm: {actor_grad_norm:.3f}  "
                f"token_level_rm_lr: {token_rm_cur_lr if args.token_level_rm is not None else 0.0:.3e}  "
                f"token_rm_grad_norm: {token_rm_grad_norm if args.token_level_rm is not None else 0.0:.3f}  "
                f"avg_judger_reward: {step_avg_judger_reward:.8f}  "
                f"avg_adv: {step_avg_adv:.8f}  "
                f"avg_gen_entropy: {step_avg_gen_entropy:.3f}  "
                f"avg_ref_kl: {step_avg_ref_kl:.8f}  "
                f"positive_loss: {step_positive_loss:.3f}  "
                f"negative_loss: {step_negative_loss:.3f}  "
                f"kl_penalty_loss: {step_kl_penalty_loss:.3f}  "
                f"rl_loss: {step_rl_loss:.3f}  "
                f"token_level_rm_loss: {step_token_level_rm_loss if args.token_level_rm is not None else 0.0:.3f}  "
                f"kl_coef: {args.kl_coef:.5f}  "
                f"max_memory: {(max_memory / 1024**3):.1f}GB  "
                f"avg_new_tokens: {int(step_avg_new_tokens)}  "
                f"num_rl_tokens: {int(step_rl_consumed_tokens)}  "
                f"infer_tgs: {int(infer_tgs)}  "
                f"rl_tgs: {int(rl_tgs)}  "
                f"gen_time: {step_gen_time:.2f}s  "
                f"infer_time: {step_infer_time:.2f}s  "
                f"rl_time: {step_rl_time:.2f}s  "
                f"total_time: {step_time:.2f}s  "
                f"eta: {eta}"
            )

        if is_interval(step, total_steps, checkpoint_interval):
            DEVICE_MODULE.empty_cache()

            num_digits = len(str(abs(total_steps)))
            work_dir = args.work_dir
            ckpt_dir = os.path.join(work_dir, f"ckpt-{step+1:0{num_digits}}")
            hf_dir = os.path.join(work_dir, f"hf-{step+1:0{num_digits}}")

            with profile_time_and_memory("[Checkpoint]"):
                actor_model.save_pretrained(hf_dir)
                tokenizer.save_pretrained(hf_dir)

                dist.barrier()

    train_cost_time = time.time() - start_train_t
    logger.success(f"[Train] Cost {timedelta(seconds=int(train_cost_time))}")
    # ------------------------    Training  End  ---------------------------- #


if __name__ == "__main__":
    fire.Fire(train_oreal)
