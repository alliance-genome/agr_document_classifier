import math
import re
from typing import List

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import PreTrainedModel, PretrainedConfig

from abc_utils import EntityType, get_all_curated_entities

OPENING_REGEX_STR = "[\\.\\n\\t\\'\\/\\(\\)\\[\\]\\{\\}:;\\,\\!\\?> ]"
CLOSING_REGEX_STR = "[\\.\\n\\t\\'\\/\\(\\)\\[\\]\\{\\}:;\\,\\!\\?> ]"


class AllianceStringMatchingEntityExtractorConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "alliance_string_matching_entity_extractor"


class AllianceStringMatchingEntityExtractor(PreTrainedModel):
    config_class = AllianceStringMatchingEntityExtractorConfig

    def __init__(self, config, mod_abbreviation, entity_type, min_matches, tfidf_threshold,
                 match_uppercase: bool = False):
        super().__init__(config)
        self.config = config
        self.mod_abbreviation = mod_abbreviation
        self.entity_type = entity_type
        self.tfidf_threshold = tfidf_threshold
        self.match_uppercase = match_uppercase
        self.min_matches = min_matches
        self.vectorizer = TfidfVectorizer()

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.custom_entity_extraction(input_ids)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

    def custom_entity_extraction(self, input_ids):
        # Decode input_ids to text
        text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        # Extract meaningful entities
        keywords = self.extract_meaningful_entities_by_keywords(
            keywords=self.get_alliance_curated_entities(self.entity_type, self.mod_abbreviation),
            text=text,
            match_uppercase=self.match_uppercase,
            min_matches=self.min_matches,
            tfidf_threshold=self.tfidf_threshold,
            blacklist=None
        )

        # Convert keywords to logits
        logits = self.keywords_to_logits(keywords, text)
        return logits

    def keywords_to_logits(self, keywords, text):
        # Initialize logits with zeros
        logits = torch.zeros(len(text), self.config.num_labels)

        # Set logits for the positions of the keywords
        for keyword in keywords:
            start_idx = 0
            while start_idx != -1:
                start_idx = text.find(keyword, start_idx)
                if start_idx != -1:
                    end_idx = start_idx + len(keyword)
                    logits[start_idx:end_idx, 1] = 1  # Assuming label 1 for entities
                    start_idx = end_idx
        return logits

    @staticmethod
    def get_alliance_curated_entities(entity_type: str, mod_abbreviation: str):
        # Fetch entities from the alliance API
        # Placeholder function
        return get_all_curated_entities(mod_abbreviation=mod_abbreviation, entity_type_str=entity_type)

    @staticmethod
    def match_entities_regex(text, regex):
        res = re.findall(regex, " " + text + " ")
        return ["".join(entity_arr) for entity_arr in res]

    @staticmethod
    def count_keyword_matches_regex(keyword, text, case_sensitive: bool = True, match_uppercase: bool = False) -> int:
        keyword = keyword if case_sensitive else keyword.upper()
        text = text if case_sensitive else text.upper()
        match_uppercase = False if keyword.upper() == keyword else match_uppercase
        if keyword in text or match_uppercase and keyword.upper() in text:
            try:
                match_count = len(re.findall(OPENING_REGEX_STR + re.escape(keyword) + CLOSING_REGEX_STR, text))
                if match_uppercase:
                    match_count += len(
                        re.findall(OPENING_REGEX_STR + re.escape(keyword.upper()) + CLOSING_REGEX_STR, text))
                return match_count
            except:
                pass
        return 0

    def is_entity_meaningful(self, entity_keywords: List[str], text, match_uppercase: bool = False,
                             min_num_occurrences: int = 1, tfidf_threshold: float = 0.0) -> bool:
        min_num_occurrences = 1 if min_num_occurrences < 1 else min_num_occurrences
        raw_count = sum(
            self.count_keyword_matches_regex(keyword=keyword, text=text, match_uppercase=match_uppercase) for keyword in
            entity_keywords)
        return True if raw_count >= min_num_occurrences and (
                    tfidf_threshold <= 0 or 0 < tfidf_threshold < self.tfidf(entity_keywords=entity_keywords,
                                                                             raw_count=raw_count, text=text)) else False

    def tfidf(self, entity_keywords: List[str], raw_count, text) -> float:
        # Fit the vectorizer on the text corpus
        self.vectorizer.fit([text])
        doc_counter = sum(self.vectorizer.vocabulary_.get(keyword, 0) for keyword in entity_keywords)
        idf = math.log(float(len(self.vectorizer.vocabulary_)) / (doc_counter if doc_counter > 0 else 0.5))
        return raw_count * idf

    def extract_meaningful_entities_by_keywords(self, keywords: List[str], text: str, match_uppercase: bool = False,
                                                min_matches: int = 1, tfidf_threshold: float = 0.0,
                                                blacklist: List[str] = None) -> List[str]:
        blacklist = set(blacklist) if blacklist else set()
        return [keyword for keyword in set(keywords) if
                keyword not in blacklist and self.is_entity_meaningful(entity_keywords=[keyword], text=text,
                                                                       match_uppercase=match_uppercase,
                                                                       min_num_occurrences=min_matches,
                                                                       tfidf_threshold=tfidf_threshold)]
