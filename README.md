# LoRe: Personalizing LLMs via Low-Rank Reward Modeling

**[LoRe](https://arxiv.org/abs/2504.14439)** is a lightweight, modular codebase for **learning personalized reward models** from preference data in multi-user environments. It supports both joint reward learning and few-shot personalization, and is built with extensibility in mind.

LoRe currently supports experiments on three benchmark datasets:
- **Reddit TLDR**: headline preference modeling
- **PRISM**: multi-turn dialogue response preferences
- **PersonalLLM**: user-personalized reward modeling for open-ended language model responses

---

## 🚀 Features

- Low-rank reward model learning across users
- Few-shot personalization with new users
- Evaluation on seen/unseen users and prompts
- Modular dataset and optimizer configuration

---

## 📦 Installation

LoRe requires Python 3.8+ and PyTorch. To install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🗂 Project Structure

```
LoRe/
├── utils.py                    # Core training, optimization, and evaluation helpers
├── RedditTLDR/                 # TLDR dataset scripts
│   ├── prepare.py              # Preprocess the dataset
│   ├── train_basis.py          # Train shared reward model and user weights
│   └── vary_fewshot.py         # Evaluate few-shot personalization
├── PRISM/                      # PRISM dataset scripts
│   ├── prepare.py
│   ├── train_basis.py
│   └── vary_fewshot.py
├── PersonalLLM/                # PersonalLLM dataset scripts
│   ├── prepare.py
│   ├── train_basis.py
```

---

## 🔧 Core Components (`utils.py`)

The following functions are central to training and evaluating personalized reward models. You may want to modify these if you’re extending LoRe:

### Model 
- `LoRe(...)`: Class modeling shared reward model `V` (linear transformation on fixed embeddings) and user-specific weights `W`
- `PersonalizeBatch(...)`: Class to model weights for new users

## Training and Evaluation
- `run(...)`: Runs the entire pipeline with 1. Learning the basis rewards, 2. Evaluation on seen users, 3. Fewshot learning on new users, 4. Evaluation on new users. The input K_list can be modified to specify the number of basis. 0 is the reference model, and 1 is the BT model.


---

## 🧪 Usage Instructions

Below are instructions for each dataset folder, following a consistent workflow:
1. **Prepare the dataset** if required.
2. **Train the reward model basis** using joint learning.
3. **Evaluate few-shot personalization** with unseen users.

### 🔹 Reddit TLDR

Inside the `RedditTLDR/` directory:
- `prepare.py`: preprocesses the TLDR dataset
- `train_basis.py`: trains the shared reward model and user weights
- `vary_fewshot.py`: evaluates few-shot personalization performance

Example usage:
```bash
cd LoRe/RedditTLDR
python prepare.py          # only needed once
python train_basis.py
python vary_fewshot.py
```

---

### 🟣 PersonalLLM

Inside the `PersonalLLM/` directory:
- `prepare.py`: prepares model response data and user splits
- `train_basis.py`: learns reward basis across users
- `vary_fewshot.py`: evaluates few-shot generalization

Example usage:
```bash
cd LoRe/PersonalLLM
python prepare.py          # only needed once
python train_basis.py
python vary_fewshot.py
```

---

### 🔸 PRISM

Inside the `PRISM/` directory:
- `prepare.py`: prepares PRISM dialogue data and embeddings
- `train_basis.py`: runs reward model training and evaluation
- `vary_fewshot.py`: runs experiments varying the number of few-shot examples

Example usage:
```bash
cd LoRe/PRISM
python prepare.py          # only needed once
python train_basis.py
python vary_fewshot.py
```
---

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
CC-BY-NC 4.0 licensed, as found in the LICENSE file.

## Cite Us
If you use this codebase, please cite us:
```bibtex
@misc{bose2025lorepersonalizingllmslowrank,
      title={LoRe: Personalizing LLMs via Low-Rank Reward Modeling}, 
      author={Avinandan Bose and Zhihan Xiong and Yuejie Chi and Simon Shaolei Du and Lin Xiao and Maryam Fazel},
      year={2025},
      eprint={2504.14439},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.14439}, 
}