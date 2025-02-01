<div align="center">
<a href="https://www.python.org/">
<img src="./docs/images/logo.svg" width="200" alt="logo"/>
</a>
<h1>RadioLLM: Introducing Large Language Model into Cognitive Radio via Hybrid Prompt and Token Reprogrammings</h1>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8-blue"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/Pytorch-latest-orange"></a>
<a href="https://arxiv.org/abs/2501.17888"><img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv-B31B1B"></a>
<a href="https://huggingface.co/datasets/"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-🤗-FFFDF5"></a>
<a href="https://github.com/SparkZu/RadioLLM"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SparkZu/RadioLLM"></a>
</div>

## 📚 Introduction
[RadioLLM](https://github.com/SparkZu/RadioLLM), a novel framework that incorporates Hybrid Prompt and Token Reprogramming (HPTR) and a Frequency Attuned Fusion (FAF) module to enhance LLMs for CRT tasks. HPTR enables the integration of radio signal features with expert knowledge, while FAF improves the modeling of high-frequency features critical for precise signal processing. These innovations allow RadioLLM to handle diverse CRT tasks, bridging the gap between LLMs and traditional signal processing methods. Extensive empirical studies on multiple benchmark datasets demonstrate that the proposed RadioLLM achieves superior performance over current baselines.

## 🔥 NEWS
- **[2025-02-01]** 📝 The preprint of the RadioLLM paper is available on arXiv. Check the [paper page](https://arxiv.org/abs/2501.17888) for more details.

## 📅 TODO
- [ ] Collect the codes of RadioLLM's classification network and other comparison models.

## 💻 Requirements

The code is implemented in Python 3.8. The required packages are listed in the `requirements.txt` file. You can install the required packages by running the following command:

```
conda create --name radiollm python=3.8
conda activate radiollm
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/peft
```

## 📖 Citation
Please cite the following paper if you use this study in your research:

```
@misc{chen2025radiollmintroducinglargelanguage,
      title={RadioLLM: Introducing Large Language Model into Cognitive Radio via Hybrid Prompt and Token Reprogrammings}, 
      author={Shuai Chen and Yong Zu and Zhixi Feng and Shuyuan Yang and Mengchang Li and Yue Ma and Jun Liu and Qiukai Pan and Xinlei Zhang and Changjun Sun},
      year={2025},
      eprint={2501.17888},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2501.17888}, 
}
```
