# OREAL: Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning


[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2405.20315-b31b1b.svg)](https://arxiv.org/abs/2405.20315)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-OREAL-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/internlm/OREAL-32B)


## ✨ Introduction

![main_fig](./figures/main_fig.jpg)

Reasoning abilities, especially those for solving complex math problems, are crucial components of general intelligence.
Recent advances by proprietary companies, such as o-series models of OpenAI, have made remarkable progress on reasoning tasks. However, the complete technical details remain unrevealed, and the techniques that are believed certainly to be adopted are only reinforcement learning (RL) and the long chain of thoughts.

We proposes a new RL framework, termed OREAL, to pursue the performance limit that can be achieved through **O**utcome **RE**w**A**rd-based reinforcement **L**earning for mathematical reasoning tasks, where only binary outcome rewards are easily accessible.

+ We theoretically prove that behavior cloning on positive trajectories from best-of-N (BoN) sampling is sufficient to learn the KL-regularized optimal policy in binary feedback environments.
+ This formulation further implies that the rewards of negative samples should be reshaped to ensure the gradient consistency between positive and negative samples.
+ To alleviate the long-existing difficulties brought by sparse rewards in RL, which are even exacerbated by the partial correctness of the long chain of thought for reasoning tasks, we further apply a token-level reward model to sample important tokens in reasoning trajectories for learning.

The OREAL implementation pseudocode is as follows:

![algo](./figures/algo.png)


## 📃 Key Results

With OREAL, for the first time, a 7B model can obtain 94.0 pass@1 accuracy on MATH-500 through RL, being on par with 32B models. OREAL-32B also surpasses previous 32B models trained by distillation with 95.0 pass@1 accuracy on MATH-500.

![main_table](./figures/main_table.png)

## 🤗 HuggingFace Model Zoo

Our OREAL models are available on Hugging Face 🤗:

| Model    | Huggingface Repo |
|----------|------------------|
| OREAL-7B  | [Model Link](https://huggingface.co/internlm/OREAL-7B)  |
| OREAL-32B  | [Model Link](https://huggingface.co/internlm/OREAL-32B)  |

We also release the models of SFT version. You can construct your own RL pipeline on them:)

| Model    | Huggingface Repo |
|----------|------------------|
| OREAL-7B-SFT  | [Model Link](https://huggingface.co/internlm/OREAL-7B-SFT)  |
| OREAL-32B-SFT  | [Model Link](https://huggingface.co/internlm/OREAL-32B-SFT)  |

## 🖊️ Citation

```

```

## 💳 License

This project is released under the Apache 2.0 [license](./LICENSE).