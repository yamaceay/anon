import os
os.environ["OMP_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer, InputExample
import faiss
import numpy as np
import torch
import shap
import re

from io import StringIO
import json
from collections import OrderedDict
from typing import List, Set, Tuple

import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class TextAnalyzer:
    def __init__(self, model: SentenceTransformer, special_tokens: List[str], stop_words: List[str]):
        self.model = model
        self.tokenizer = self.model.tokenizer

        self.special_tokens = special_tokens
        self.stopwords = stop_words

        self.shap_masker = shap.maskers.Text(self.tokenizer)
        self.shap_explainer = shap.Explainer(self.model.encode, masker=self.shap_masker, show_progress_bar=False)
    
    def get_token_offsets(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
        tokens = [text[start:end] for start, end in offsets]
        return tokens, offsets

    def find_special_token_spans(self, text: str) -> List[Tuple[int, int]]:
        special_token_spans = []
        for special in self.special_tokens:
            for match in re.finditer(re.escape(special), text):
                special_token_spans.append((match.start(), match.end()))
        return special_token_spans

    def get_special_token_indices(self, offsets: List[Tuple[int, int]], 
                                special_token_spans: List[Tuple[int, int]]) -> Set[int]:
        special_token_indices = set()
        for idx, (start, end) in enumerate(offsets):
            for span_start, span_end in special_token_spans:
                if start >= span_start and end <= span_end:
                    special_token_indices.add(idx)
        return special_token_indices

    def is_valid_token(self, token: str) -> bool:
        if token in self.stopwords:
            return False
        if not token.isalpha() and not token.startswith("##"):
            return False
        if re.match(r"^\W+$", token):
            return False
        return True

    def process_tokens(self, tokens: List[str], shap_values: np.ndarray, 
                      special_token_indices: Set[int]) -> List[Tuple[str, float]]:
        filtered_token_weights = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if i in special_token_indices:
                i += 1
                continue
                
            if token.startswith("##"):
                if filtered_token_weights:
                    prev_token, prev_weight = filtered_token_weights.pop()
                    merged_token = prev_token + token[2:]
                    filtered_token_weights.append((merged_token, prev_weight))
            else:
                clean_token = token.replace("##", "")
                if self.is_valid_token(clean_token):
                    filtered_token_weights.append((clean_token, shap_values[i]))
            i += 1
        return filtered_token_weights

    def analyze_text(self, text: str) -> List[Tuple[str, float]]:
        shap_values = self.explain(text)
        tokens, offsets = self.get_token_offsets(text)
        special_token_spans = self.find_special_token_spans(text)
        special_token_indices = self.get_special_token_indices(offsets, special_token_spans)
        return self.process_tokens(tokens, shap_values, special_token_indices)

    def explain(self, text):
        token_weights = self.shap_explainer([text])
        # tokens = token_weights.data[0]
        weights = token_weights.values[0]
        if len(weights.shape) > 1:
            weights = weights.sum(axis=-1)

        return weights

class PETRE:
    def __init__(self, input_file, model_name, output_dir="outputs/petre", device=device):
        self.model_name = model_name
        self.input_file = input_file
        self.output_dir = output_dir
        self.device = device

        self.individual_name_column = "name"
        self.background_knowledge_column = "public_knowledge"
        self.selected_anonymized_column = "spacy_abstract"
        self.explainer_name = "shap"
        self.mask_token = "[MASK]"

    def load_pretreated_pairs(self):
        with open(self.input_file, "r") as f:
            (train_df_json_str, eval_dfs_jsons) = json.load(f)        
            
        train_list = json.loads(StringIO(train_df_json_str).read())
        pairs = []
        for row in train_list:
            name = row.pop(self.individual_name_column)
            public_knowledge = row.pop(self.background_knowledge_column)
            pairs.append(InputExample(texts=[name, public_knowledge], label=1.0))

        eval_pairs_dict = {}
        eval_dict = OrderedDict([(name, json.loads(StringIO(df_json).read())) for name, df_json in eval_dfs_jsons.items()])
        for key, eval_list in eval_dict.items():
            for row in eval_list:
                name = row.pop(self.individual_name_column)
                anonymized_knowledge = row.pop(key)
                if key not in eval_pairs_dict:
                    eval_pairs_dict[key] = []
                eval_pairs_dict[key].append(InputExample(texts=[name, anonymized_knowledge], label=1.0))

        return pairs, eval_pairs_dict
    
    def build_index(self):
        print("Building FAISS index...")
        dim = self.doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(self.doc_embeddings)
        return index

    def build(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.spacy_nlp = en_core_web_lg.load()
        self.masked_values = set()
        for label in self.spacy_nlp.get_pipe("ner").labels:
            self.masked_values.add(label)
        self.masked_values = list(self.masked_values)
        self.stopwords = set(self.spacy_nlp.Defaults.stop_words)

        self.explainer = None
        if self.explainer_name == "shap":
            self.explainer = TextAnalyzer(self.model, self.masked_values, self.stopwords)

        _, self.data_dfs = self.load_pretreated_pairs()
        self.data_pairs = self.data_dfs[self.selected_anonymized_column]

        self.pairs = [ex for ex in self.data_pairs if ex.label == 1.0]
        self.names = [ex.texts[0] for ex in self.pairs]
        self.docs = [ex.texts[1] for ex in self.pairs]

        self.doc_embeddings = self.model.encode(
            self.docs,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=4,
        )
        self.index = self.build_index()

    def save_index(self, path):
        faiss.write_index(self.index, path)
        print(f"FAISS index saved to {path}.")

    def load_index(self, path):
        self.index = faiss.read_index(path)
        print(f"FAISS index loaded from {path}.")

    def retrieve_top_k(self, queries, k=5, show_progress_bar=True):
        print(f"Retrieving top-{k} similar documents...")
        query_emb = self.model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )

        results = []
        dists, idxs = self.index.search(query_emb, k)
        for dist_row, idx_row in zip(dists, idxs):
            scores = np.exp(-dist_row)  # negative distance = higher score
            scores = scores / scores.sum()  # normalize to probabilities
            result = [
                {"doc": self.docs[idx], "score": float(score)}
                for idx, score in zip(idx_row, scores)
            ]
            results.append(result)
        return results

    def compute_rank(self, text, doc, rank_k):
        [retrievals] = self.retrieve_top_k([text], k=rank_k, show_progress_bar=False)
        rank = 1
        for item in retrievals:
            if item["doc"] == doc:
                break
            rank += 1
        
        if rank <= rank_k:
            score = retrievals[rank - 1]["score"]
            score_diff = [item["score"] - score for item in retrievals[rank:]]
            print(f"Rank: {rank}, Score: {score:.4f}, Score Diff: {score_diff}")
        return rank

    def mask_text_by_token(self, text, token, weight, num_masked=None):
        if num_masked is not None:
            text = text.replace(token, self.mask_token, num_masked)
            print(f"Replaced {num_masked} '{token}' -> {self.mask_token} (importance={weight:.4f}).")
        else:
            text = text.replace(token, self.mask_token)
            print(f"Replaced all '{token}' -> {self.mask_token} (importance={weight:.4f}).")

        return text

    def run(self, idx, rank_k, num_tokens=1, num_masked=None):
        doc, name = self.docs[idx], self.names[idx]
        print(f"\n--- Processing Document {idx+1} ---")
        text = doc
        while True:
            rank = self.compute_rank(text, doc, rank_k)
            if rank > rank_k:
                print(f"Document escaped risk zone! (rank={rank})")
                break
            
            explanation = self.explainer.analyze_text(text)
            sorted_explanation = reversed(sorted(explanation, key=lambda x: abs(x[1])))

            num_tokens_left = num_tokens
            for token, weight in sorted_explanation:
                if num_tokens_left <= 0:
                    print("No more tokens to mask.")
                    break # No more tokens to mask
                
                if not token:
                    continue
                
                text = self.mask_text_by_token(text, token, weight, num_masked)

                num_tokens_left -= 1

            if num_tokens_left > 0:
                break # No more tokens to mask

        # write the results to a file
        output_file = os.path.join(self.output_dir, f"masked_doc_{idx+1}.txt")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(f"Entity: {name}\n")
            f.write(f"Original Document:\n{doc}\n")
            f.write(f"Masked Document:\n{text}\n")
            f.write(f"Rank: {rank}\n")
        return text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PETRE for risk evaluation.")
    parser.add_argument("--input_file", "-i", type=str, default="data/wiki_toy_done.json",
                        help="Path to the pretreated data pairs JSON file.")
    parser.add_argument("--model_dir", "-s", type=str, default="outputs/sbert_model",
                        help="Directory containing the pre-trained SentenceTransformer model.")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/petre",
                        help="Directory to save the PETRE results.")
    args = parser.parse_args()

    petre = PETRE(input_file=args.input_file, model_name=args.model_dir, output_dir=args.output_dir)
    petre.build()

    print("\nRunning PETRE iterative masking...")
    masked_doc = petre.run(idx=4, rank_k=2, num_masked=1)