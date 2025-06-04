import json

import torch
from torch.utils.data import DataLoader
from io import StringIO
from collections import OrderedDict
import faiss
import random
import os
import numpy as np

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

random.seed(42)  # Set random seed for reproducibility

device = (
    torch.device("cuda") if torch.cuda.is_available() else 
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

class Retriever:
    def __init__(self, model, pairs):
        self.model = model
        self.pairs = [ex for ex in pairs if ex.label == 1.0]
        [self.people, self.docs] = list(zip(*[pair.texts for pair in self.pairs]))
        doc_emb = self.model.encode(self.docs, convert_to_numpy=True, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(doc_emb.shape[1])
        self.index.add(doc_emb)

    def retrieve(self, queries, k=5):
        query_emb = self.model.encode(queries, convert_to_numpy=True)
        dists, idxs = self.index.search(query_emb, k)
        return [(self.people[i], self.docs[i], dists[0][rank]) for rank, i in enumerate(idxs[0])]
    
def get_evaluator(pairs, prefix):
    queries = {f'q{i}': ex.texts[0] for i, ex in enumerate(pairs)}
    corpus = {f'd{i}': ex.texts[1] for i, ex in enumerate(pairs)}
    relevant_docs = {f'q{i}': [f'd{i}'] for i in range(len(pairs))}
    evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=prefix)
    return evaluator

class SBERTTrainer:
    def __init__(self, input_file, checkpoint_dir):
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.individual_name_column = 'name'
        self.background_knowledge_column = 'public_knowledge'
        self.selected_anonymized_columns = ['spacy_abstract']

        self.checkpoint_dir = checkpoint_dir
        self.input_file = input_file

        self.eval_data_for_training = False
        self.train_model = True
        self.save_finetuned_model = True
        self.validate_model = True
        self.load_finetuned_model = False

        self.num_negatives=10
        self.negatives_min_sim=0.2
        self.negatives_max_sim=0.6

        self.training_args = SentenceTransformerTrainingArguments(
            output_dir=self.checkpoint_dir,
            save_strategy="steps",           # Save every N steps
            save_steps=500,                  # Save every 500 steps
            save_total_limit=3,              # Keep only the 3 most recent/best models
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=25,
            eval_strategy="steps",     # Evaluate every N steps
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="map",     # Must match the metric name in your evaluator
            greater_is_better=True,
            seed=42)

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
                name = row.pop('name')
                anonymized_knowledge = row.pop(key)
                if key not in eval_pairs_dict:
                    eval_pairs_dict[key] = []
                eval_pairs_dict[key].append(InputExample(texts=[name, anonymized_knowledge], label=1.0))

        return pairs, eval_pairs_dict

    def sample_semi_hard_negatives(self):
        """Sample semi-hard negatives using the current model."""
        all_texts = [ex.texts[1] for ex in self.train_pairs]
        embeddings = self.model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
        new_pairs = []
        for pair in self.train_pairs:
            query_emb = self.model.encode([pair.texts[0]], convert_to_numpy=True, show_progress_bar=False)
            sims = embeddings @ query_emb.T / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) * np.linalg.norm(query_emb)
            )
            sims = sims.squeeze()
            sims = [(i, sim) for i, sim in enumerate(sims) if all_texts[i] != pair.texts[1]]
            filtered = [idx for idx, sim in sims if self.negatives_min_sim < sim < self.negatives_max_sim]
            if not filtered:
                continue
            sampled_neg_idx = random.choice(filtered[:self.num_negatives])  # Choose top-k or randomly
            negative_text = all_texts[sampled_neg_idx]
            new_pairs.append(InputExample(texts=[pair.texts[0], negative_text], label=0.0))
        return new_pairs

    def run(self):
        self.train_pairs, self.eval_pairs_dict = self.load_pretreated_pairs()

        if self.load_finetuned_model and os.path.exists(self.checkpoint_dir):
            self.transformer = Transformer(self.checkpoint_dir)
            self.pooling_model = Pooling(self.transformer.get_word_embedding_dimension(), "mean")
            self.model = SentenceTransformer(modules=[self.transformer, self.pooling_model], device=device)
        else:
            self.model = SentenceTransformer(self.model_name, device=device)

        semi_hard_negatives = self.sample_semi_hard_negatives()
        self.train_pairs.extend(semi_hard_negatives)

        if self.eval_data_for_training:
            for column, eval_pairs in self.eval_pairs_dict.items():
                if column in self.selected_anonymized_columns:
                    self.train_pairs.extend(eval_pairs)

        print(f"Before training, evaluating model on training data...")
        evaluator = get_evaluator(self.train_pairs, "train")
        evaluation_results = self.model.evaluate(evaluator)

        print(f"Before training, evaluating model on evaluation data...")
        if self.validate_model and not self.eval_data_for_training:
            selected_eval_pairs = []
            for key, eval_pairs in self.eval_pairs_dict.items():
                if key in self.selected_anonymized_columns:
                    selected_eval_pairs.extend(eval_pairs)
            evaluator = get_evaluator(selected_eval_pairs, "eval")
            
        if self.train_model:
            # train_loss = losses.Co sineSimilarityLoss(model) # not suitable for multiple negatives
            self.train_loss = losses.MultipleNegativesRankingLoss(self.model)
            self.train_dataloader = DataLoader(self.train_pairs, shuffle=True, batch_size=16)

            self.model.fit(
                train_objectives=[(self.train_dataloader, self.train_loss)],
                evaluator=evaluator,
                epochs=int(self.training_args.num_train_epochs),
                warmup_steps=self.training_args.warmup_steps,
                optimizer_params={'lr': self.training_args.learning_rate},
                evaluation_steps=self.training_args.eval_steps,
                output_path=self.training_args.output_dir,
                save_best_model=self.training_args.load_best_model_at_end,
                checkpoint_save_total_limit=self.training_args.save_total_limit,
            )

        print(f"After training, evaluating model on training data...")
        evaluator = get_evaluator(self.train_pairs, "train")
        evaluation_results = self.model.evaluate(evaluator)

        print(f"After training, evaluating model on evaluation data...")
        if self.validate_model and not self.eval_data_for_training:
            evaluator = get_evaluator(selected_eval_pairs, "eval")
            evaluation_results = self.model.evaluate(evaluator)
            
        if self.save_finetuned_model:
            self.model.save(self.checkpoint_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model for RAG.")
    parser.add_argument('--input-file', '-i', type=str, default='data/wiki_toy_done.json', help='Path to the input data file.')
    parser.add_argument('--checkpoint-dir', '-o', type=str, default='outputs/sbert_model', help='Directory to save the model checkpoints.')
    args = parser.parse_args()

    sbert_trainer = SBERTTrainer(args.input_file, args.checkpoint_dir)
    sbert_trainer.run()

    print("Testing the retriever...")
    retriever = Retriever(sbert_trainer.model, sbert_trainer.train_pairs)
    
    query = "Who is an American mixed martial artist?"
    retrieved_docs = retriever.retrieve([query], k=5)
    print(f"Query: {query}")
    for ppl, doc, dist in retrieved_docs:
        print(f"Distance: {dist:.4f}, Entity: {ppl}, Document: {doc}")