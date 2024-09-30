<h1> Fine-Tuning Llama Models for Sentiment Analysis </h1>

This repository contains two Jupyter notebooks for fine-tuning Llama models (Llama 2 and Llama 3.2.1B) for sentiment analysis on financial and economic data. The sentiment analysis focuses on helping businesses gain valuable insights into market trends, investor confidence, consumer behavior, risk management, and investment decision-making.

Notebooks Overview
1. Fine-tune-llama-2-for-sentiment-analysis.ipynb: A step-by-step guide to fine-tuning the Llama 2 model for sentiment analysis tasks.
2. Fine-tune-llama-3.2.1B-for-sentiment-analysis.ipynb: A similar guide but using the more advanced Llama 3.2.1B model for sentiment analysis.


### Parameter-Efficient Fine-Tuning (PEFT) Parameters

The **PEFTConfig** object specifies the parameters used for Parameter-Efficient Fine-Tuning (PEFT):

- **`lora_alpha`**: The learning rate for the LoRA update matrices.
- **`lora_dropout`**: The dropout probability for the LoRA update matrices.
- **`r`**: The rank of the LoRA update matrices, determining the dimension of low-rank matrix approximations.
- **`bias`**: Specifies the type of bias to use. Possible values are:
  - `none`: No bias is used.
  - `additive`: An additive bias is applied.
  - `learned`: Bias is learned during training.
- **`task_type`**: The type of task for which the model is being trained (e.g., text classification, language modeling).

---

### Simple Fine-Tuning Trainer (SFTTrainer) Parameters

The **SFTTrainer** is a custom trainer class used to train large language models with PEFT. The key parameters used for **SFTTrainer** are:

- **`model`**: The pre-trained Llama model to be fine-tuned.
- **`train_dataset`**: The dataset used for training.
- **`eval_dataset`**: The dataset used for evaluation.
- **`peft_config`**: Configuration of the PEFT method (as described above).
- **`dataset_text_field`**: The name of the text field in the dataset (e.g., `input_text`).
- **`tokenizer`**: The tokenizer used to process text data for the model.
- **`args`**: Training arguments and hyperparameters:
  - **`output_dir`**: Directory where training logs and checkpoints will be saved.
  - **`num_train_epochs`**: The number of epochs to train the model.
  - **`per_device_train_batch_size`**: The number of samples in each batch per device.
  - **`gradient_accumulation_steps`**: The number of steps to accumulate gradients before updating model weights.
  - **`optim`**: The optimizer used for training (e.g., `AdamW`).
  - **`save_steps`**: The number of steps after which a checkpoint is saved.
  - **`logging_steps`**: The number of steps after which training metrics are logged.
  - **`learning_rate`**: The learning rate used by the optimizer.
  - **`weight_decay`**: Weight decay (regularization) for the optimizer to reduce overfitting.
  - **`fp16`** and **`bf16`**: Whether to use mixed precision (`16-bit`) training for faster performance.
  - **`max_grad_norm`**: Maximum gradient norm used for gradient clipping.
  - **`max_steps`**: The maximum number of steps for which to train the model.
  - **`warmup_ratio`**: The proportion of training steps used for warming up the learning rate.
  - **`group_by_length`**: Whether to group the training samples by length for efficient batching.
  - **`lr_scheduler_type`**: Type of learning rate scheduler to adjust the learning rate during training.
  - **`report_to`**: The tool used for reporting training metrics (e.g., TensorBoard).
  - **`evaluation_strategy`**: Defines how and when to evaluate the model during training.

---
