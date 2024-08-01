import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig
import random
from tqdm import tqdm


class RLHF:
    def __init__(self, base_model_id: str):

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model_id, 
            load_in_4bit=True,
            peft_config=lora_config,
            device_map='balanced'
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ppo_config = PPOConfig(
                model_name="mistral-orkg-synthesis",
                learning_rate=1.41e-5,
                batch_size=1,
                mini_batch_size=1
        )
        self.ppo_trainer = PPOTrainer(
            model=self.model,
            config=self.ppo_config,
            tokenizer=self.tokenizer,
        )

        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 200,
            "batch_size":1,
        }
    
    def train(self, dataset, epochs: int=1):

        for epoch in tqdm(range(epochs), "epoch: "):
            for data in tqdm(dataset): 
                synthesis_type, query = data['synthesis'], data['prompt']
                query_tensors = self.tokenizer(query, return_tensors='pt').to("cuda")
            
                query_tensors = query_tensors.input_ids[0]
                
                response_tensors = self.ppo_trainer.generate(query_tensors,  **self.generation_kwargs)
                
                query_response = [self.tokenizer.decode(response[len(query_tensors):]) for response in response_tensors]
            
                #### Compute reward score
                # texts = [q + r for q, r in zip(query, query_response)]
                # pipe_outputs = reward_model(texts)
                
                rewards = [torch.tensor(random.uniform(0, 3)) for output in query_response]
            
                #### Run PPO step
                stats = self.ppo_trainer.step([query_tensors], list(response_tensors), rewards)
                # ppo_trainer.log_stats(stats, batch, rewards)
    
    def save(self, path):
        self.ppo_trainer.save_pretrained(path)



class SFT:
    pass


class SFTRLHF:
    pass
