# -*- coding: utf-8 -*-
from scripts import io
from scripts.configs import BaseConfig
from scripts.dataset import SynthesisDatasetBuilder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import os
from transformers import StoppingCriteria, StoppingCriteriaList

access_token = os.environ['HUGGINGFACE_ACCESS_TOKEN']


class MaxWordsStoppingCriteria(StoppingCriteria):
    def __init__(self, max_words, tokenizer):
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).split("###\n\n [/INST]")[1]
        words = text.split()
        return len(words) >= self.max_words and text[-1] == '.'


def inference(prompt, is_llama, stopping_criteria_list):
    messages = [
        # {"role": "system", "content": "Follow the user instructions to the letter. The output should be only what the user requests, do not respond with anything else."},
        {"role": "user", "content": prompt}
    ]

    encodeds = tokenizer.apply_chat_template(messages, temperature=0, return_tensors="pt")

    model_inputs = encodeds.to(device)

    if is_llama:
        generated_ids = model.generate(model_inputs,
                                       pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id,
                                       max_new_tokens=400,
                                       temperature=0.7,
                                       top_k=20,
                                       top_p=0.95,
                                       do_sample=True,
                                       num_return_sequences=1)
    else:
        generated_ids = model.generate(model_inputs,
                                       pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id,
                                       max_new_tokens=400,
                                       temperature=0.7,
                                       top_k=20,
                                       top_p=0.95,
                                       do_sample=True,
                                       num_return_sequences=1,
                                       stopping_criteria=stopping_criteria_list)

    if is_llama:
        response = generated_ids[0][model_inputs.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response[0].split("###\n\n [/INST]")[1]


if __name__ == '__main__':
    args = BaseConfig().get_args()

    if args.user_arg_inf:
        inf_configs = [
            (args.input_llm_dir, args.output_inf_path)
        ]
    else:
        inf_configs = [
            (args.llm_warmup_dir, args.llm_warmup_inf),                           # SFT
            (args.rlhf_style_warmup_dir, args.rlhf_style_warmup_inf),             # SFT + RL (w/Basic Features)
            (args.rlhf_style_dir, args.rlhf_style_inf),                           # RL (w/Basic Features)
            (args.rlhf_style_gpt4_warmup_dir, args.rlhf_style_gpt4_warmup_inf),   # SFT + RL (w/Basic Features) + RLAIF (w/GPT4 Features)
            (args.rlhf_style_gpt4_dir, args.rlhf_style_gpt4_inf),                 # RL (w/Basic Features) + RLAIF (w/GPT4 Features)
        ]

    for llm_dir, inf_path in inf_configs:
        df = io.read_csv(args.orkg_synthesis_test)
        print("size of the dataset is: ", df.shape[0])
        df.head(2)

        dataset_builder = SynthesisDatasetBuilder(df=df,
                                                  prompt_template=args.synthesis_prompt_template,
                                                  synthesis_type_dict=args.synthesis_type_dict)
        test_data = dataset_builder.orkg_synthesis_inf()
        print("datase size:", len(test_data))
        print(f"Inferencing using {llm_dir} model!")

        device = "cuda"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(llm_dir,
                                                     quantization_config=bnb_config,
                                                     device_map={"": 0})
        padding_side = 'left' if args.is_llama else "right"
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, padding_side=padding_side, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token

        desired_word_count = 200
        stopping_criteria = MaxWordsStoppingCriteria(max_words=desired_word_count, tokenizer=tokenizer)
        stopping_criteria_list = StoppingCriteriaList([stopping_criteria])

        if not os.path.exists(inf_path):
            for index in tqdm(range(len(test_data))):
                inf = inference(test_data[index]['prompt'], args.is_llama, stopping_criteria_list)
                print("Size of the text is:", len(inf.split()))
                test_data[index]['inference'] = inf
                io.write_json(output_path=inf_path, json_data=test_data)
