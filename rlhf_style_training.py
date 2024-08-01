# args.base_model_id (in model)
# args.rlhf_style_optimizer_name
# args.rlhf_style_dir

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig
import random
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
import os
from transformers.optimization import Adafactor
from datasets import Dataset

from scripts import io
from scripts.dataset import SynthesisDatasetBuilder
from scripts.configs import BaseConfig
from scripts.reward import BasciFeaturesRewardModel

import warnings
warnings.simplefilter("ignore")


_ = load_dotenv(find_dotenv())

access_token = os.environ['HUGGINGFACE_ACCESS_TOKEN']

def build_dataset(dataset, tokenizer, input_max_text_length):
    def tokenize(sample):
        prompt = sample["prompt"]
        messages = [{"role": "user", "content": prompt}]
        sample["input_ids"] = tokenizer.apply_chat_template(messages, temperature=0, return_tensors="pt")[0][: input_max_text_length]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    ds = Dataset.from_list(dataset)
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

if __name__ == "__main__":
    args = BaseConfig().get_args()

    print("args.rlhf_style_optimizer_name:", args.rlhf_style_optimizer_name)
    print("args.base_model_id:", args.base_model_id)
    print("args.rlhf_style_dir:", args.rlhf_style_dir)
    print("---"*30)

    df = io.read_csv(args.orkg_synthesis_train_rlhf) 
    
    print("size of the dataset is: ", df.shape[0])
    print(df.columns)

    dataset_builder = SynthesisDatasetBuilder(df=df, 
                                              prompt_template=args.synthesis_prompt_template, 
                                              synthesis_type_dict=args.synthesis_type_dict)

    train_data = dataset_builder.orkg_synthesis_rlhf()
    print("datase size:", len(train_data))

    reward_model = BasciFeaturesRewardModel(args=args)
    reward_model.build_reward_model()

    peft_config = LoraConfig(
        r=args.lora_config_r,
        lora_alpha=args.lora_config_lora_alpha,
        lora_dropout=args.lora_config_lora_dropout,
        target_modules=args.lora_config_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model_id, 
        load_in_4bit=True,
        peft_config=peft_config,
        device_map='balanced',
        bnb_4bit_compute_dtype=torch.float16
    )
    if args.optimizer_type == 'adafactor':
        optimizer=Adafactor(model.parameters(), lr=args.ppo_learning_rate, relative_step=False)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.ppo_learning_rate)
    
    padding_side = 'left' if  args.is_llama else "right"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, padding_side=padding_side, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()

    dataset = build_dataset(dataset=train_data, tokenizer=tokenizer, input_max_text_length=args.input_max_text_length)

    config = PPOConfig(
        model_name=args.rlhf_style_optimizer_name,
        learning_rate=args.ppo_learning_rate,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.ppo_batch_size,
        mini_batch_size=args.ppo_mini_batch_size,
        gradient_accumulation_steps=args.ppo_gradient_accumulation_steps,
        log_with='tensorboard',
        project_kwargs={'logging_dir':args.rlhf_style_dir} ,
    )

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
        num_shared_layers=4,
        
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": args.max_token_len,
        # "batch_size":1,
        # "eos_token_id": tokenizer.eos_token_id,
    }
    rewards_lst = []
    train_step = 0
    for epoch in range(args.rlhf_num_train_epochs):
        for batch_no, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]

            print(f"Epoch:{epoch}/{args.rlhf_num_train_epochs} -- Batch No: {batch_no}/{len(train_data)} -- step: {train_step}")
            
            response_tensors = []
            for query in query_tensors:
                while True:
                    response = ppo_trainer.generate(query, return_prompt=False, **generation_kwargs)
                    if len(response[0]) > 10:
                        break
                # print("Generation size:", len(response[0]))
                response_tensors.append(response.squeeze())
                
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            
            rewards = [torch.tensor(float(reward_model.get_reward(output))) for output in batch["response"]]
            for output in batch["response"]:
                rewards_lst.append(reward_model.get_reward(output))

            print(f"REWARD: {rewards} -- AVG:{sum(rewards_lst)/len(rewards_lst)}")

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            batch['batch-no'] = f"Epoch:{epoch}/{args.rlhf_num_train_epochs} -- Batch No: {batch_no}/{len(train_data)} -- step: {train_step}"
            ppo_trainer.log_stats(stats, batch, rewards)
            
            torch.cuda.empty_cache()
            
            if train_step%100 == 0:
                print("Run ppo_trainer.save_pretrained!")
                ppo_trainer.save_pretrained(os.path.join(args.rlhf_style_dir, f"step-{str(train_step)}"))
            train_step += 1

            print("--"*40)

        print("Run ppo_trainer.save_pretrained!")
        ppo_trainer.save_pretrained(os.path.join(args.rlhf_style_dir, f"epoch-{str(epoch)}"))
    ppo_trainer.save_pretrained(args.rlhf_style_dir)
