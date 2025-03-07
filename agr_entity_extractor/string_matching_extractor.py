from collections import defaultdict

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import PreTrainedModel, PretrainedConfig


class AllianceStringMatchingEntityExtractorConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "alliance_string_matching_entity_extractor"
        # Configure your labels as needed.
        self.id2label = {0: "O", 1: "ENTITY"}
        self.label2id = {"O": 0, "ENTITY": 1}


# Custom model for token classification using string matching.
class AllianceStringMatchingEntityExtractor(PreTrainedModel):
    config_class = AllianceStringMatchingEntityExtractorConfig

    def __init__(self, config, entity_type, min_matches, tfidf_threshold,
                 tokenizer, entities_to_extract, match_uppercase: bool = False):
        super().__init__(config)
        self.config = config
        self.tfidf_threshold = tfidf_threshold
        self.match_uppercase = match_uppercase
        self.min_matches = min_matches
        self.tokenizer = tokenizer
        self.entities_to_extract = set(entities_to_extract)
        # Dummy parameter so that the model has parameters.
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.custom_entity_extraction(input_ids)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

    def custom_entity_extraction(self, input_ids):
        """
        Produce logits at token level.
        The logits tensor should have shape (batch_size, seq_length, num_labels).
        """
        batch_tokens = [self.tokenizer.convert_ids_to_tokens(seq) for seq in input_ids]
        logits_list = []
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None
        )
        tfidf_vectorizer.fit(batch_tokens)

        global_token_counts = defaultdict(int)
        for tokens in batch_tokens:
            for token in tokens:
                if token in self.entities_to_extract:
                    global_token_counts[token] += 1

        for tokens in batch_tokens:
            # Initialize token-level logits: shape (num_tokens, num_labels).
            token_logits = torch.zeros(len(tokens), self.config.num_labels, device=input_ids.device)
            # Get the TF-IDF values for the document.
            doc_tfidf = tfidf_vectorizer.transform([tokens])
            # For each token in the document...
            for i, token in enumerate(tokens):
                if token in self.entities_to_extract:
                    # Use the in-document frequency (count) for this token.
                    token_count = global_token_counts[token]
                    # Get the tf-idf score for this token, if it exists in the fitted vocabulary.
                    if token in tfidf_vectorizer.vocabulary_:
                        feature_index = tfidf_vectorizer.vocabulary_[token]
                        tfidf_value = doc_tfidf[0, feature_index]
                    else:
                        tfidf_value = self.tfidf_threshold
                    # Check if the token meets both the frequency and TF-IDF threshold criteria.
                    if token_count >= self.min_matches and (
                            self.tfidf_threshold <= 0 or tfidf_value > self.tfidf_threshold):
                        token_logits[i, 1] = 1.0  # Label 1 for ENTITY detected.
            logits_list.append(token_logits)
            # Return a tensor of shape (batch_size, seq_length, num_labels).
        return torch.stack(logits_list, dim=0)
