# Model Related Settings
actor = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
reference = actor
token_level_rm = actor

# Tokenizer related settings
# jinja2 template for hf tokenizer
chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
stop_word = "<｜end▁of▁sentence｜>"

dtype = "auto"
selective_recompute = 1.0
cpu_offload = False
cuda_graph = True
tp_size = 4
sp_size = 1

# Dataset Related Settings
data_difficulty_balance_cfg = [
    # pass rate range, repeat times
    ((0.0, 0.2), 6),
    ((0.2, 0.4), 4),
    ((0.4, 0.6), 4),
    ((0.6, 0.8), 2),
]
datasets = "internlm/OREAL-RL-Prompts"
num_workers = 0

# Generate Related Settings
gen_global_batch = 1024
gen_max_new = 14000
gen_max_length = 16384
gen_top_k = 0  # set to 0 means not use topk sampling
gen_top_p = 0.9
temperature = 1.0
gen_do_sample = True
max_prefill_batch = 16
prompt_repeat_k = 16  # sample k times for each prompt

# Optimizer Related Settings
rl_global_batch = gen_global_batch
rl_mirco_batch = 2
filter_trajectory = True  # sample one correct and one incorrect trajectory for each prompt
warmup_steps = 10
total_steps = 90
actor_freeze_steps = 10  # freeze actor and only update token level reward model for the first 10 steps
actor_lr = 5e-7
actor_min_lr = 1e-7
token_level_rm_lr = 2e-6
token_level_rm_lr_min = 4e-7
wd = 0.01  # weight decay
max_grad_norm = 1  # gradient clipping

# importance sampling setting with token level reward model
threshold_rescale = True
correct_threshold = 0.5
incorrect_threshold = 0.5
# topk_rescale = True
# correct_topk_ratio = 0.25
# incorrect_topk_ratio = 0.25

reward_shaping_type = "rloo"
loss_type = "per_token"
positive_loss_factor = 1.0
negative_loss_factor = 0.5
pos_mult_adv = True
kl_coef = 0.01  # KL coefficient

# General Settings
work_dir = "work_dirs"  # directory to save logs and checkpoints
checkpoint_interval = 10  # interval to save checkpoint, <1 means save by proportion, >=1 means save by steps
log_interval = 1  # interval steps for logging
seed = 0  # random seed
debug = False  # set log level to DEBUG

# judger related settings
judgers_config = dict(
    math_judger=dict(  # math judger related settings
        hosts=[
            "YOUR_JUDGER_HOST1:PORT",
            "YOUR_JUDGER_HOST2:PORT",
        ],
        stop_word=stop_word,
        thinking_finish_words=["<conclude>", "**Final Answer**", "</think>"],
        num_processes=8,
        concurrency_per_proc=(8, 8),
    )
)
data_judger_mapping = dict(math=["math_judger"])
