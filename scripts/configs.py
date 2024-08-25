# -*- coding: utf-8 -*-
"""
    BaseConfig: Data Configuration of models
"""
import argparse
import os
from pathlib import Path
from typing import Dict
from scripts import io

# import openai
# from dotenv import find_dotenv, load_dotenv

# _ = load_dotenv(find_dotenv())


class BaseConfig:
    def __init__(self):
        self.root_dataset_dir = Path(__file__).parents[1]
        self.parser = argparse.ArgumentParser()

    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def get_args(self, device="cpu", batch_size: int = None, nshots: int = None) -> Dict:
        self.device = device
        self.batch_size = batch_size
        prompt_template = io.read_text(os.path.join(self.root_dataset_dir, "templates/standardized_synthesis_generation.txt"))
        orkg_synthesis_train = os.path.join(self.root_dataset_dir, "dataset/split/ORKG_Synthesis_train.csv")
        orkg_synthesis_test = os.path.join(self.root_dataset_dir, "dataset/split/ORKG_Synthesis_test.csv")
        orkg_synthesis_train_llm = os.path.join(self.root_dataset_dir, "dataset/split/ORKG_Synthesis_train_llm.csv")
        orkg_synthesis_train_rlhf = os.path.join(self.root_dataset_dir, "dataset/split/ORKG_Synthesis_train_rlhf.csv")

        synthesis_type_dict = {
            'methodological': io.read_text(input_path=os.path.join(self.root_dataset_dir, "templates/objectives/methodological.txt")),
            'thematic': io.read_text(input_path=os.path.join(self.root_dataset_dir, "templates/objectives/thematic.txt")),
            'paperwise': io.read_text(input_path=os.path.join(self.root_dataset_dir, "templates/objectives/paperwise.txt"))
        }

        # General configurations
        self.parser.add_argument("--root_dir", type=str, default=self.root_dataset_dir)
        self.parser.add_argument("--orkg_synthesis_train", type=str, default=orkg_synthesis_train)
        self.parser.add_argument("--orkg_synthesis_test", type=str, default=orkg_synthesis_test)
        self.parser.add_argument("--orkg_synthesis_train_llm", type=str, default=orkg_synthesis_train_llm)
        self.parser.add_argument("--orkg_synthesis_train_rlhf", type=str, default=orkg_synthesis_train_rlhf)
        self.parser.add_argument("--synthesis_prompt_template", type=str, default=prompt_template)
        self.parser.add_argument("--synthesis_type_dict", type=dict, default=synthesis_type_dict)

        # LLM
        self.parser.add_argument("--base_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
        self.parser.add_argument("--llm_num_train_epochs", type=int, default=5)
        self.parser.add_argument("--llm_warmup_dir", type=str, default=os.path.join(self.root_dataset_dir, 'assets/warmup'))
        self.parser.add_argument("--llm_warmup_inf", type=str, default=os.path.join(self.root_dataset_dir, 'assets/warmup-inf.json'))
        self.parser.add_argument("--max_token_len", type=int, default=400)

        # RLHF
        self.parser.add_argument("--reward_vocab", type=str, default=os.path.join(self.root_dataset_dir, 'scripts/reward-vocab.txt'))
        self.parser.add_argument("--rlhf_style_optimizer_name", type=str, default='mistral-orkg-synthesis-rlhf-style')
        self.parser.add_argument("--rlhf_style_dir", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style'))
        self.parser.add_argument("--rlhf_style_inf", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-inf.json'))
        self.parser.add_argument("--rlhf_num_train_epochs", type=int, default=1)

        # finetuning + RLHF (basic)
        self.parser.add_argument("--rlhf_style_warmup_dir", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-with-warmup'))
        self.parser.add_argument("--rlhf_style_warmup_inf", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-with-warmup-inf.json'))
        self.parser.add_argument("--rlhf_style_warmup_optimizer_name", type=str, default='mistral-orkg-synthesis-rlhf-style-with-warmup')

        # finetuning + RLHF (basic+gpt4)
        self.parser.add_argument("--rlhf_style_gpt4_warmup_start_model", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-with-warmup/epoch-0'))
        self.parser.add_argument("--rlhf_style_gpt4_warmup_dir", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-gpt4-with-warmup'))
        self.parser.add_argument("--rlhf_style_gpt4_warmup_eval_loger", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-gpt4-with-warmup/eval-loger.json'))
        self.parser.add_argument("--rlhf_style_gpt4_warmup_inf", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-gpt4-with-warmup-inf.json'))
        self.parser.add_argument("--rlhf_style_gpt4_warmup_optimizer_name", type=str, default='mistral-orkg-synthesis-rlhf-style-gpt4-with-warmup')

        # RLHF (basic+gpt4)
        self.parser.add_argument("--rlhf_style_gpt4_start_model", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style'))
        self.parser.add_argument("--rlhf_style_gpt4_dir", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-gpt4'))
        self.parser.add_argument("--rlhf_style_gpt4_eval_loger", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-gpt4/eval-loger.json'))
        self.parser.add_argument("--rlhf_style_gpt4_inf", type=str, default=os.path.join(self.root_dataset_dir, 'assets/rlhf-style-gpt4-inf.json'))
        self.parser.add_argument("--rlhf_style_gpt4_optimizer_name", type=str, default='mistral-orkg-synthesis-rlhf-style-gpt4')

        # eval
        eval1_system_prompt_problem = io.read_text(os.path.join(self.root_dataset_dir, "templates/evaluation1_system_prompt_problem.txt"))
        self.parser.add_argument("--eval1_system_prompt_problem", type=str, default=eval1_system_prompt_problem)
        eval2_system_prompt_problem = io.read_text(os.path.join(self.root_dataset_dir, "templates/evaluation2_system_prompt_problem.txt"))
        self.parser.add_argument("--eval2_system_prompt_problem", type=str, default=eval2_system_prompt_problem)
        eval2_characteristic = io.read_json(os.path.join(self.root_dataset_dir, "templates/evaluation2_characteristic.json"))
        self.parser.add_argument("--eval2_characteristic_lst", type=list, default=eval2_characteristic)

        # PPO Configs
        self.parser.add_argument("--ppo_learning_rate", type=float, default=(1.47e-5) * 2)
        self.parser.add_argument("--ppo_epochs", type=int, default=10)
        self.parser.add_argument("--ppo_batch_size", type=int, default=1)
        self.parser.add_argument("--ppo_mini_batch_size", type=int, default=1)
        self.parser.add_argument("--ppo_gradient_accumulation_steps", type=int, default=1)
        self.parser.add_argument("--optimizer_type", type=str, default='adam')
        self.parser.add_argument("--input_max_text_length", type=int, default=2950)

        # LoraConfig
        self.parser.add_argument("--lora_config_r", type=int, default=8)
        self.parser.add_argument("--lora_config_lora_alpha", type=int, default=16)
        self.parser.add_argument("--lora_config_lora_dropout", type=float, default=0.05)
        self.parser.add_argument("--lora_config_target_modules", type=list,
                                 default=["v_proj", "down_proj", "o_proj", "up_proj", "gate_proj", "q_proj", "k_proj"])

        # Allow inference to work with user args
        self.parser.add_argument("--user_arg_inf", type=bool, default=False)
        self.parser.add_argument("--input_llm_dir", type=str)
        self.parser.add_argument("--output_inf_path", type=str)
        self.parser.add_argument("--is_llama", type=bool, default=False)

        self.parser.add_argument("-f")
        return self.parser.parse_args()
