from tqdm import tqdm
from scripts import io
from scripts.configs import BaseConfig
from scripts.dataset import format_context, build_gpt4_user_prompt, build_gpt4_user_prompt_eval2
from scripts.utils import call_gpt4_api, paper_structure_score, word_count_score

import time
import os

FOLD_NO = 3

def check_gpt4_eval_key(prefix, keys):
    required_keys=[prefix+str(idx+1) for idx in range(FOLD_NO)]
    for req_key in required_keys:
        found = False
        for key in keys:
            if req_key == key:
                found=True
                break
        if not found:
            return True, req_key
    return False, ""
    

if __name__=='__main__':
    args = BaseConfig().get_args()
    inf_paths = [
        # 'assets/vanila.json', # -- result is OK
        # args.llm_warmup_inf,  # -- result is OK
        # args.rlhf_style_warmup_inf,
        # args.rlhf_style_inf,
        # args.rlhf_style_gpt4_warmup_inf, # -- result is OK
        # args.rlhf_style_gpt4_inf, # -- result is OK

        # "assets/vanilla-llama3.json",
        # "assets/warmup-inf-llama3.json",
        # "assets/rlhf-style-with-warmup-inf-llama3.json",
        # "assets/rlhf-style-gpt4-with-warmup-inf-llama3.json",
        # "assets/gpt4-vanilla.json"
        "assets/gpt4-vanilla.json"
    ]

    reward_vocab = io.read_text(args.reward_vocab).split("\n")
    reward_vocab = [vocab.lower() for vocab in reward_vocab]

    for inf_path in inf_paths:

        finetuned_inf = io.read_json(inf_path)
        df = io.read_csv(args.orkg_synthesis_test)
        print("size of the dataset is: ", len(finetuned_inf))


        for idx in tqdm(range(len(finetuned_inf))):
            if "llama" in inf_path:
                synthesis = finetuned_inf[idx]['inference'].replace("userassistant"," ").replace("useruser"," ")
            else:
                synthesis = finetuned_inf[idx]['inference']
            
            while False: # if "gpt-4-eval-s1" not in list(finetuned_inf[idx].keys()):
                do_eval, eval_key = check_gpt4_eval_key(prefix="gpt-4-eval-s1-", keys=list(finetuned_inf[idx].keys()))
                print(f"Eval-key: {eval_key}, for item {idx}")
                if not do_eval:
                    break
                row = df[df['sample_id'] == finetuned_inf[idx]['sample_id']]
                research_problem = row['research_problem'].tolist()[0]
                synthesis_type = finetuned_inf[idx]['synthesis']

                context = format_context(row)
                
                user_prompt = build_gpt4_user_prompt(synthesis=synthesis, research_problem=research_problem, synthesis_type=synthesis_type, context=context)
                
                message = [{"role": "system", "content": args.eval1_system_prompt_problem},
                           {"role": "user", "content": user_prompt}]

                response, completion_tokens, prompt_tokens = call_gpt4_api(message)

                finetuned_inf[idx][eval_key] = {
                    "eval-result": response, 
                    "completion-tokens": completion_tokens, 
                    "prompt-tokens":prompt_tokens
                }

                io.write_json(output_path=inf_path, json_data=finetuned_inf)

            while False: # if "gpt-4-eval-s2" not in list(finetuned_inf[idx].keys()):
                do_eval, eval_key = check_gpt4_eval_key(prefix="gpt-4-eval-s2-", keys=list(finetuned_inf[idx].keys()))
                print(f"Eval-key: {eval_key}, for item {idx}")
                if not do_eval:
                    break
                row = df[df['sample_id'] == finetuned_inf[idx]['sample_id']]
                research_problem = row['research_problem'].tolist()[0]
                synthesis_type = finetuned_inf[idx]['synthesis']
                
                context = format_context(row)

                user_prompt = build_gpt4_user_prompt_eval2(synthesis=synthesis, research_problem=research_problem, synthesis_type=synthesis_type, context=context)
                
                responses = {}
                completion_tokens = 0
                prompt_tokens = 0

                for charactrestic_dict in args.eval2_characteristic_lst:
                    system_prompt = args.eval2_system_prompt_problem
                    for key, val in charactrestic_dict.items():
                        system_prompt = system_prompt.replace(key, val)

                    message = [{"role": "system", "content": system_prompt}, 
                               {"role": "user", "content": user_prompt}]

                    response, completion_token, prompt_token = call_gpt4_api(message, max_tokens=150)
                    responses[charactrestic_dict['<characteristic-title>']] = response[list(response.keys())[0]]
                    completion_tokens += completion_token
                    prompt_tokens += prompt_token

                finetuned_inf[idx][eval_key] = {
                    "eval-result": responses, 
                    "completion-tokens": completion_tokens, 
                    "prompt-tokens":prompt_tokens
                }

                io.write_json(output_path=inf_path, json_data=finetuned_inf)

            if "basic-eval" not in list(finetuned_inf[idx].keys()):
                finetuned_inf[idx]['basic-eval'] = {
                    "paper-structure": paper_structure_score(synthesis, reward_vocab), 
                    "word-count": word_count_score(synthesis)
                }
                io.write_json(output_path=inf_path, json_data=finetuned_inf)