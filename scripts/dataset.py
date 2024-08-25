# -*- coding: utf-8 -*-
from datasets import Dataset


def format_context(row):
    context = ""
    for i in range(5):
        title = row[f'paper_{i+1}_title'].tolist()[0]
        abstract = row[f'paper_{i+1}_abstract'].tolist()[0]
        # context += f'{i+1}. ' + ' '.join(title.replace('\n', ' ').split()) + '\n' + ' '.join(abstract.replace('\n', ' ').split()) + '\n\n'
        context += f'{i+1}. ' + title + '\n' + abstract.replace('\n', ' ') + '\n\n'
    return context


def format_context_training(row):
    context = ""
    for i in range(5):
        title = row[f'paper_{i+1}_title']
        abstract = row[f'paper_{i+1}_abstract']
        # context += f'{i+1}. ' + ' '.join(title.replace('\n', ' ').split()) + '\n' + ' '.join(abstract.replace('\n', ' ').split()) + '\n\n'
        context += f'{i+1}. ' + title + '\n' + abstract.replace('\n', ' ') + '\n\n'
    return context


def fill_prompt(row, prompt_template, synthesis_instruct):
    papers = format_context(row).strip()
    problem = row['research_problem'].tolist()[0]
    prompt = prompt_template.format(research_problem=problem, synthesis_type=synthesis_instruct)
    prompt += '\n\nPapers:\n' + papers + '\n\n###\n\n'
    return prompt


def fill_prompt_training(row, prompt_template, synthesis_instruct):
    papers = format_context_training(row).strip()
    problem = row['research_problem']
    prompt = prompt_template.format(research_problem=problem, synthesis_type=synthesis_instruct)
    prompt += '\n\nPapers:\n' + papers + '\n\n###\n\n'
    return prompt


def build_gpt4_user_prompt(synthesis, research_problem, synthesis_type, context):
    instruct = 'Evaluate and rate the quality of the following scientific synthesis according to the nine characteristics given in the system prompt.'
    user_prompt = instruct + f'''\n\n<scientific-synthesis>{synthesis}<\\scientific-synthesis>\n\n<research-problem>{research_problem}<\\research-problem>\n
<synthesis-type>{synthesis_type}<\\synthesis-type>\n\n<paper-titles-and-abstracts>{context}<\\paper-titles-and-abstracts>\n\n###'''
    # user_prompt = instruct + f'''\n\n<scientific-synthesis>\n{synthesis}\n<\\scientific-synthesis>\n\n<research-problem>\n{research_problem}\n<\\research-problem>\n
    # \n<synthesis-type>\n{synthesis_type}<\\synthesis-type>\n\n<paper-titles-and-abstracts>\n{context}\n<\\paper-titles-and-abstracts>\n\n###'''
    return user_prompt


def build_gpt4_user_prompt_eval2(synthesis, research_problem, synthesis_type, context):
    instruct = 'Evaluate and rate the quality of the following scientific synthesis according to the characteristic given in the system prompt.'
    user_prompt = instruct + f'''\n\n<scientific-synthesis>\n{synthesis}\n<\\scientific-synthesis>\n\n<research-problem>\n{research_problem}\n<\\research-problem>\n
<synthesis-type>\n{synthesis_type}\n<\\synthesis-type>\n\n<paper-titles-and-abstracts>\n{context}\n<\\paper-titles-and-abstracts>\n\n###'''
    return user_prompt


class SynthesisDatasetBuilder:

    def __init__(self, df, prompt_template, synthesis_type_dict):
        self.df = df
        self.prompt_template = prompt_template
        self.synthesis_type_dict = synthesis_type_dict

    def orkg_synthesis_rlhf(self):
        dataset = []
        for index, row in self.df.iterrows():
            for synthesis_type, synthesis_instruct in self.synthesis_type_dict.items():
                prompt = fill_prompt_training(row, self.prompt_template, synthesis_instruct)
                dataset.append({"synthesis": synthesis_type, "prompt": prompt})
        return dataset

    def orkg_synthesis_inf(self):
        dataset = []
        for index, row in self.df.iterrows():
            for synthesis_type, synthesis_instruct in self.synthesis_type_dict.items():
                prompt = fill_prompt_training(row, self.prompt_template, synthesis_instruct)
                dataset.append({"synthesis": synthesis_type, "split": row['split'], "sample_id": row['sample_id'], "prompt": prompt})
        return dataset

    def orkg_synthesis_rlhf_gpt4(self):
        dataset = []
        for index, row in self.df.iterrows():
            context = format_context_training(row).strip()
            research_problem = row['research_problem']

            for synthesis_type, synthesis_instruct in self.synthesis_type_dict.items():
                prompt = fill_prompt_training(row, self.prompt_template, synthesis_instruct)
                # if synthesis_type == "paperwise":
                #     synthesis_type = "paper-wise"
                dataset.append({"synthesis": synthesis_type, "prompt": prompt, "context": context, "research_problem": research_problem})
        return dataset

    def orkg_synthesis_reward(self):
        dataset = []
        for index, row in self.df.iterrows():
            for synthesis_type, synthesis_instruct in self.synthesis_type_dict.items():
                synthesis = row[f"mistral_{synthesis_type}_synthesis"]
                reward = row[f"mistral_{synthesis_type}_paper_output_format"]
                if str(reward) == "nan":
                    reward = 1
                else:
                    reward = 0
                dataset.append({"synthesis-type": synthesis_type, "synthesis": synthesis, "reward": reward})
        return dataset

    def orkg_synthesis_llm(self, is_llama=False):
        dataset = []
        for index, row in self.df.iterrows():
            for synthesis_type, synthesis_instruct in self.synthesis_type_dict.items():
                prompt = fill_prompt_training(row, self.prompt_template, synthesis_instruct)
                output_text = row[f'gpt4_{synthesis_type}_synthesis']
                if is_llama:
                    PT = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|> {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|> {output_text} <|eot_id|>"""
                else:
                    PT = f"""<s>[INST] {prompt} [/INST] {output_text}</s>"""
                dataset.append({
                    "synthesis": synthesis_type,
                    "prompt": prompt,
                    "output": output_text,
                    "prompt-template": PT
                })
        return Dataset.from_list(dataset)
