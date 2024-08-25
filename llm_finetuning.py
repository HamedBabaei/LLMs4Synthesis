# -*- coding: utf-8 -*-
from scripts import io
from scripts.dataset import SynthesisDatasetBuilder
from scripts.configs import BaseConfig

import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from dotenv import find_dotenv, load_dotenv
import os
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

_ = load_dotenv(find_dotenv())


def find_all_linear_names(model):
    # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        # needed for 16-bit
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


if __name__ == '__main__':
    access_token = os.environ['HUGGINGFACE_ACCESS_TOKEN']
    args = BaseConfig().get_args()

    print("args.llm_warmup_dir:", args.llm_warmup_dir)
    print("args.base_model_id:", args.base_model_id)
    print("---" * 30)

    df = io.read_csv(args.orkg_synthesis_train_llm)

    print("size of the dataset is: ", df.shape[0])
    print(df.columns)

    dataset_builder = SynthesisDatasetBuilder(df=df,
                                              prompt_template=args.synthesis_prompt_template,
                                              synthesis_type_dict=args.synthesis_type_dict)
    train_data = dataset_builder.orkg_synthesis_llm(is_llama=args.is_llama)
    print("datase size:", len(train_data))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(args.base_model_id, quantization_config=bnb_config, device_map={"": 0}, token=access_token)
    padding_side = 'left' if args.is_llama else "right"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, add_eos_token=True, padding_side=padding_side, token=access_token)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)
    print('linear layers for fine tuning:', modules)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        dataset_text_field="prompt-template",
        peft_config=lora_config,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=0.03,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=args.llm_warmup_dir,
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            num_train_epochs=args.llm_num_train_epochs
        ),
        max_seq_length=args.max_token_len,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    trainer.model.save_pretrained(args.llm_warmup_dir)
