import os
os.environ["OMP_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer, InputExample
import faiss
import numpy as np
import torch
import shap

from io import StringIO
import json
from collections import OrderedDict

import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class PETRE:
    def __init__(self, input_file, model_name):
        self.model_name = model_name
        self.input_file = input_file
        self.device = device

        self.individual_name_column = "name"
        self.text_column = "text"
        self.selected_anonymized_column = "spacy_abstract"
        self.explainer_name = "shap"

    def load_pretreated_pairs(self):
        with open(self.input_file, "r") as f:
            (train_df_json_str, eval_dfs_jsons) = json.load(f)        
            
        train_list = json.loads(StringIO(train_df_json_str).read())
        pairs = []
        for row in train_list:
            name = row.pop('name')
            public_knowledge = row.pop('public_knowledge')
            pairs.append(InputExample(texts=[name, public_knowledge], label=1.0))

        eval_pairs_dict = {}
        eval_dict = OrderedDict([(name, json.loads(StringIO(df_json).read())) for name, df_json in eval_dfs_jsons.items()])
        for key, eval_list in eval_dict.items():
            for row in eval_list:
                name = row.pop('name')
                anonymized_knowledge = row.pop(key)
                if key not in eval_pairs_dict:
                    eval_pairs_dict[key] = []
                eval_pairs_dict[key].append(InputExample(texts=[name, anonymized_knowledge], label=1.0))

        return pairs, eval_pairs_dict
    
    def get_spacy_labels(self):
        labels = set()
        for label in self.spacy_nlp.get_pipe("ner").labels:
            labels.add(label)
        return list(labels)
    
    def get_special_chars(self):
        special_tokens = ["[MASK]", "[CLS]", "[SEP]"]
        special_chars = [" ", "\n", "\t", ".", ",", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}", "'", "\""]
        return special_tokens + special_chars

    def build_index(self):
        print("Building FAISS index...")
        dim = self.doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(self.doc_embeddings)
        return index

    def build(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.masker = shap.maskers.Text(self.model.tokenizer)
        self.explainer = None
        if self.explainer_name == "shap":
            self.explainer = shap.Explainer(
                self.model.encode, 
                feature_names=None, 
                masker=self.masker, 
                show_progress_bar=False,
            )

        _, self.data_dfs = self.load_pretreated_pairs()
        self.data_pairs = self.data_dfs[self.selected_anonymized_column]

        self.pairs = [ex for ex in self.data_pairs if ex.label == 1.0]
        self.names = [ex.texts[0] for ex in self.pairs]
        self.docs = [ex.texts[1] for ex in self.pairs]

        self.spacy_nlp = en_core_web_lg.load()
        self.masked_values = set(self.get_spacy_labels() + self.get_special_chars())

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

    def explain(self, text):
        token_weights = self.explainer([text])
        tokens = token_weights.data[0]
        weights = token_weights.values[0]
        if len(weights.shape) > 1:
            weights = weights.sum(axis=-1)

        return zip(tokens, weights)

    def mask_token(self, text, token, weight, num_masked=None):
        if not token.strip() or token.strip() in self.masked_values:
            return text, False
        
        if num_masked is not None:
            text = text.replace(token, "[MASK]", num_masked)
            print(f"Replaced {num_masked} '{token}' -> [MASK] (importance={weight:.4f}).")
        else:
            text = text.replace(token, "[MASK]")
            print(f"Replaced all '{token}' -> [MASK] (importance={weight:.4f}).")

        return text, True

    def run(self, idx, rank_k, num_tokens=1, num_masked=None):
        doc, name = self.docs[idx], self.names[idx]
        print(f"\n--- Processing Document {idx+1} ---")
        print(f"Entity: {name}, Original Document:\n{doc[:60]}\n{'-'*40}")

        text = doc
        while True:
            rank = self.compute_rank(text, doc, rank_k)
            if rank > rank_k:
                print(f"Document escaped risk zone! (rank={rank})")
                break

            explanation = self.explain(text)
            sorted_explanation = reversed(sorted(explanation, key=lambda x: abs(x[1])))

            num_tokens_left = num_tokens
            for token, weight in sorted_explanation:
                if num_tokens_left <= 0:
                    break # No more tokens to mask
                
                text, masked = self.mask_token(text, token, weight, num_masked)
                if not masked:
                    continue

                num_tokens_left -= 1

            if num_tokens_left > 0:
                break # No more tokens to mask

        print(f"Entity: {name}, Masked Document:\n{text[:60]}\n{'-'*40}")
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

    petre = PETRE(input_file=args.input_file, model_name=args.model_dir)
    petre.build()

    print("\nRunning PETRE iterative masking...")
    masked_doc = petre.run(idx=4, rank_k=2)