# -*- coding: utf-8 -*-
from openai import OpenAI
import os
from dotenv import find_dotenv, load_dotenv
import re
import time

_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_gpt4_api(message, max_tokens=700):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=message,
                temperature=0.0,
                max_tokens=max_tokens,
                stop=['###'],
                seed=10
            )
            return eval(response.choices[0].message.content), response.usage.completion_tokens, response.usage.prompt_tokens,
        except Exception as exc:
            print(f"something went wrong! going for 5 second sleep!: {exc}")
            time.sleep(5)


def identify_references(text, log=False):
    reference_patterns = [
        r"\((19|20)\d{2}\)",
        r"https?://\S+",
        r"doi:\S+",
        r"[A-Z][a-z]+, [A-Z]\." ,
        r"\(\d+:\s[^;]+(?:;\s\d+:\s[^;]+)*\)",
        r"\(\d+\)\s\[[^\]]+\]",
        r'\(\d+\)\s+[A-Z][a-zA-Z. ]+(?: et al\.)?,',
        r'\(\d+\) [^;]+;',
    ]
    for pattern in reference_patterns:
        if len(re.findall(pattern, text)) != 0:
            if log:
                print(pattern, re.findall(pattern, text))
            return 1
    pattern = r'^\(\d+\) .+$'
    if len(re.findall(pattern, text, re.MULTILINE)) != 0:
        return 1
    return 0


def identify_paper_sections(synthesis, reward_vocab, log=False):
    contains = []
    for vocab in reward_vocab:
        if vocab in synthesis.lower():
            if log:
                print(vocab)
            contains.append(vocab)
            break
    return len(contains)


def paper_structure_score(synthesis, reward_vocab):
    if identify_references(synthesis) == 1:
        return 1
    elif identify_paper_sections(synthesis, reward_vocab) == 1:
        return 1
    else:
        return 0


def word_count_score(synthesis):
    pattern = r'\(\d+(?:[-,]\s?\d+)*\)'
    synthesis = re.sub(pattern, '', synthesis.replace(".", "")).strip()
    return {
        "count": len(synthesis.split()),
        "WC>200": 1 if len(synthesis.split()) > 200 else 0,
        "WC<50": 1 if len(synthesis.split()) < 50 else 0,
        "50<=WC<150": 1 if len(synthesis.split()) < 150 and len(synthesis.split()) >= 50 else 0,
        "150<=WC<=220": 1 if len(synthesis.split()) <= 220 and len(synthesis.split()) >= 150 else 0,
        "220<=WC<=250": 1 if len(synthesis.split()) <= 250 and len(synthesis.split()) >= 220 else 0,
        "150<=WC<=250": 1 if len(synthesis.split()) <= 250 and len(synthesis.split()) >= 150 else 0,
        "WC>250": 1 if len(synthesis.split()) > 250 else 0,
    }
