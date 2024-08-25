# -*- coding: utf-8 -*-
from scripts import io
from scripts.dataset import SynthesisDatasetBuilder, build_gpt4_user_prompt
from scripts.utils import paper_structure_score, call_gpt4_api, word_count_score

from nltk import word_tokenize
from nltk.tokenize import MWETokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Any


class ManualVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        features = []
        for text in data:
            text_lenght = len(text.split())
            features.append([text_lenght])
        return features


class StylisticRewardModel:

    def __init__(self, args):
        self.args = args
        self.reward_model = None
        self.is_reward_model_finetuned = False

    def build_reward_model(self) -> Any:
        df = io.read_csv(self.args.orkg_synthesis_train)
        dataset_builder = SynthesisDatasetBuilder(df=df,
                                                  prompt_template=self.args.synthesis_prompt_template,
                                                  synthesis_type_dict=self.args.synthesis_type_dict)
        train_data = dataset_builder.orkg_synthesis_reward()
        # rewards = {0:0, 1:0}
        # X_synthesis, y_reward = [], []
        # for data in train_data:
        #     X_synthesis.append(data['synthesis'])
        #     y_reward.append(data['reward'])
        X_synthesis, y_reward = [], []
        for data in train_data:
            if len(data['synthesis'].split()) > 250:
                y_reward.append(0)
            else:
                y_reward.append(data['reward'])
            X_synthesis.append(data['synthesis'])

        reward_vocab = io.read_text(self.args.reward_vocab).split("\n")
        reward_vocab = [vocab.lower().replace(":", " :") for vocab in reward_vocab]
        dictionary = [tuple(vocab.split(' ')) for vocab in reward_vocab]
        dictionary_tokenizer = MWETokenizer(dictionary, separator=' ')
        vectorizer = CountVectorizer(vocabulary=reward_vocab,
                                     lowercase=True,
                                     token_pattern=None,
                                     # sublinear_tf=True,
                                     # use_idf=False,
                                     tokenizer=lambda text: dictionary_tokenizer.tokenize(word_tokenize(text)))
        manual_feature_pipeline = Pipeline(steps=[("costum_vect", ManualVectorizer())])
        self.reward_model = Pipeline(steps=[
            ("Stylistic Representation", FeatureUnion(transformer_list=[('Paper Format Detector', vectorizer),
                                                                        ('Lenght Analyzer', manual_feature_pipeline)])),
            ('RewardClassifier', LogisticRegression())
        ])
        self.reward_model.fit(X_synthesis, y_reward)
        self.is_reward_model_finetuned = True

    def get_model(self):
        if not self.is_reward_model_finetuned:
            self.build_reward_model()
        return self.reward_model

    def get_reward(self, text) -> float:
        if not self.is_reward_model_finetuned:
            self.build_reward_model()
        if len(text.split()) < 30:
            return 0.1
        elif len(text.split()) >= 30 and len(text.split()) < 70:
            return 0.2

        return self.reward_model.predict_proba([text])[0][1]


class BasciFeaturesRewardModel:

    def __init__(self, args):
        self.args = args

    def build_reward_model(self) -> Any:
        reward_vocab = io.read_text(self.args.reward_vocab).split("\n")
        self.reward_vocab = [vocab.lower() for vocab in reward_vocab]
        self.is_reward_model_finetuned = True

    def get_model(self):
        return self

    def get_reward(self, text) -> float:
        if not self.is_reward_model_finetuned:
            self.build_reward_model()
        wc = word_count_score(text)['count']
        if wc < 50:
            return -1.5
        if paper_structure_score(synthesis=text, reward_vocab=self.reward_vocab) == 1:
            return -2
        if wc > 200 and abs(wc - 200) <= 20:
            return -0.5
        if wc > 200:
            return -1
        return 2

    def get_reward_for_gpt4(self, text):
        if not self.is_reward_model_finetuned:
            self.build_reward_model()
        wc = word_count_score(text)['count']
        if wc < 100:
            wcs = -4
        elif wc > 200 and abs(wc - 200) <= 20:
            wcs = -2
        elif wc > 200:
            wcs = -5
        else:
            wcs = 5
        pss = paper_structure_score(synthesis=text, reward_vocab=self.reward_vocab)
        return wcs, pss


class GPT4FeaturesRewardModel:

    def __init__(self, args):
        self.args = args
        self.baseic_reward_model = BasciFeaturesRewardModel(self.args)

    def build_reward_model(self) -> Any:
        self.baseic_reward_model.build_reward_model()
        self.is_reward_model_finetuned = True

    def get_model(self):
        return self

    def get_reward(self, query_response, query) -> float:
        if not self.is_reward_model_finetuned:
            self.build_reward_model()

        def preferred_value_score(numbers, preferred_value):
            return -sum(abs(x - preferred_value) for x in numbers) / len(numbers)

        user_prompt = build_gpt4_user_prompt(synthesis=query_response,
                                             research_problem=query['research_problem'],
                                             synthesis_type=query['synthesis'],
                                             context=query['context'])
        system_prompt = self.args.eval1_system_prompt_problem
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        while True:
            try:
                response, completion_tokens, prompt_tokens = call_gpt4_api(message)
                scores = [int(rating['rating']) for _, rating in response.items()]
                reward = preferred_value_score(scores, preferred_value=5)
                if reward >= -0.125:
                    reward = float(4.0)
                break
            except Exception as exc:
                print(f"ERROR while calling GPT4-API in REWARD MODELING!: {exc}")
        return {
            "reward": reward,
            "message": message,
            "wc": word_count_score(query_response)['count'],
            "response": response,
            "completion-tokens": completion_tokens,
            "prompt-tokens": prompt_tokens
        }
