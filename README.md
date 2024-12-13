# mindnlp-QNLI
A repository comparing the inference accuracy of MindNLP and Transformer on the GLUE QNLI dataset


## Quick Start

### Installation
To get started with mindnlp-QNLI, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xuhangscut/mindnlp-qnli.git
2. **Create a conda environment (optional but recommended):**
    ```bash
    conda create -n mindnlp python==3.9
    conda activate mindnlp
2. **Install the dependencies:**
Please note that mindnlp is in the Ascend environment, while transformer is in the GPU environment, and the required dependencies are in the requirements of their respective folders.
    ```bash
    cd mindnlp #or cd transformers
    pip install -r requirements.txt
3. **Usage**
Once the installation is complete, you can choose use differnet models to start inference. Here's how to run the inference:
    ```bash
    python bart-qnli.py
## Accuracy Comparsion
|  Model Name | bart | bert | roberta | xlm-roberta | gpt2 | t5 | distilbert | albert | opt | llama |
|---|---|---|---|---|---|---|---|---|---|---|
|  base Model  | facebook/bart-base  |  google-bert/bert-base-uncased | FacebookAI/roberta-large | FacebookAI/xlm-roberta-large |  openai-community/gpt2 |  google-t5/t5-small |  distilbert/distilbert-base-uncased | albert/albert-base-v2  | facebook/opt-125m  | JackFram/llama-160m  |
|  finetuning Model(hf)  | ModelTC/bart-base-qnli  | Li/bert-base-uncased-qnli  | howey/roberta-large-qnli | tmnam20/xlm-roberta-large-qnli-1 | tanganke/gpt2_qnli  | lightsout19/t5-small-qnli  | anirudh21/distilbert-base-uncased-finetuned-qnli  | orafandina/albert-base-v2-finetuned-qnli  | utahnlp/qnli_facebook_opt-125m_seed-1  | Cheng98/llama-160m-qnli  |
| transformer(GPU) |  92.29 | 67.43  | 94.50 | 92.50 | 88.15  | 89.71  | 59.21  | 55.14  | 86.10  |  50.97 |
| mindnlp(NPU) | 92.29  | 67.43  | 94.51 | 92.50 | 88.15  | 89.71  | 59.23  | 55.13  | 86.10  | 50.97  |