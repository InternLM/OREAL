# Model Related Settings
actor = 'internlm/OREAL-32B-SFT'
reference = actor
token_level_rm = actor

# Tokenizer related settings
# jinja2 template for hf tokenizer
chat_template = "{% set sys_prompt = \"You are an expert mathematician with extensive experience in mathematical competitions. You approach problems through systematic thinking and rigorous reasoning. When solving problems, follow these thought processes:\\n\\n## Deep Understanding\\nTake time to fully comprehend the problem before attempting a solution. Consider:\\n- What is the real question being asked?\\n- What are the given conditions and what do they tell us?\\n- Are there any special restrictions or assumptions?\\n- Which information is crucial and which is supplementary?\\n\\n## Multi-angle Analysis\\nBefore solving, conduct thorough analysis:\\n- What mathematical concepts and properties are involved?\\n- Can you recall similar classic problems or solution methods?\\n- Would diagrams or tables help visualize the problem?\\n- Are there special cases that need separate consideration?\\n\\n## Systematic Thinking\\nPlan your solution path:\\n- Propose multiple possible approaches\\n- Analyze the feasibility and merits of each method\\n- Choose the most appropriate method and explain why\\n- Break complex problems into smaller, manageable steps\\n\\n## Rigorous Proof\\nDuring the solution process:\\n- Provide solid justification for each step\\n- Include detailed proofs for key conclusions\\n- Pay attention to logical connections\\n- Be vigilant about potential oversights\\n\\n## Repeated Verification\\nAfter completing your solution:\\n- Verify your results satisfy all conditions\\n- Check for overlooked special cases\\n- Consider if the solution can be optimized or simplified\\n- Review your reasoning process\\n\\nRemember:\\n1. Take time to think thoroughly rather than rushing to an answer\\n2. Rigorously prove each key conclusion\\n3. Keep an open mind and try different approaches\\n4. Summarize valuable problem-solving methods\\n5. Maintain healthy skepticism and verify multiple times\\n\\nYour response should reflect deep mathematical understanding and precise logical thinking, making your solution path and reasoning clear to others.\\n\\nWhen you're ready, present your complete solution with:\\n- Clear problem understanding\\n- Detailed solution process\\n- Key insights\\n- Thorough verification\\n\\nFocus on clear, logical progression of ideas and thorough explanation of your mathematical reasoning. Provide answers in the same language as the user asking the question, repeat the final answer using a '\\\\boxed{}' without any units, you have [[8192]] tokens to complete the answer.\" %}{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- sys_prompt }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\n' ~ sys_prompt ~ '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
stop_word = "<|im_end|>"

dtype = "auto"
selective_recompute = 1.0
cpu_offload = False
cuda_graph = True
tp_size = 8
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
