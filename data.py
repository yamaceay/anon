import os
import json
import gc
import re
import logging
from collections import OrderedDict

from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
from io import StringIO
import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg)

import torch

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

class DataPreprocessor():
    def __init__(self, input_file, output_file):
        self.data_file_path = input_file
        self.pretreated_data_path = output_file
        self.individual_name_column = "name"
        self.background_knowledge_column = "public_knowledge"
        self.dev_set_column_name = "dev_abstract"
        self.load_saved_pretreatment = True
        self.add_non_saved_anonymizations = True
        self.anonymize_background_knowledge = True
        self.only_use_anonymized_background_knowledge = False
        self.use_document_curation = False
        self.save_pretreatment = True
        self.base_model_name = "distilbert-base-uncased"
        self.tokenization_block_size = 250
        
        self.data_df:pd.DataFrame = None
        self.train_df:pd.DataFrame = None
        self.eval_dfs:dict = None
        self.train_individuals:set = None
        self.eval_individuals:set = None
        self.all_individuals:set = None
        self.no_train_individuals:set = None
        self.no_eval_individuals:set = None
        self.label_to_name:dict = None
        self.name_to_label:dict = None
        self.spacy_nlp = None
        self.pretreated_data_loaded:bool = None

        # Check for GPU with CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            self.device = torch.device("cpu")

    def run(self, verbose=True):
        self.pretreated_data_loaded = False
        self.pretreatment_done = False

        if self.load_saved_pretreatment and os.path.isfile(self.pretreated_data_path):
            self.train_df, self.eval_dfs = self.load_pretreatment()            
            self.pretreated_data_loaded = True

            if self.add_non_saved_anonymizations:
                self.saved_anons = set(self.eval_dfs.keys())

                new_data_df = self.read_data()
                _, new_eval_dfs = self.split_data(new_data_df)
                self.non_pretreated_anons = set(new_eval_dfs.keys())

                self.non_saved_anons = []
                for anon_name in self.non_pretreated_anons:
                    if not anon_name in self.saved_anons:
                        self.non_saved_anons.append(anon_name)

                if len(self.non_saved_anons) > 0:
                    for anon_name in self.non_saved_anons:
                        if self.use_document_curation:
                            self.curate_df(new_eval_dfs[anon_name], self.load_spacy_nlp())
                        self.eval_dfs[anon_name] = new_eval_dfs[anon_name]
                    self.pretreatment_done = True

        else:
            self.data_df = self.read_data()
            self.train_df, self.eval_dfs = self.split_data(self.data_df)
            del self.data_df # Remove general dataframe for saving memory

        res = self.get_individuals(self.train_df, self.eval_dfs)
        self.train_individuals, self.eval_individuals, self.all_individuals, self.no_train_individuals, self.no_eval_individuals = res

        self.label_to_name, self.name_to_label, self.num_labels = self.get_individuals_labels(self.all_individuals)

        if verbose:
            self.show_data_stats(self.train_df, self.eval_dfs, self.no_eval_individuals, self.no_train_individuals, self.eval_individuals)

        if (self.anonymize_background_knowledge or self.use_document_curation) and not self.pretreated_data_loaded:
            if self.anonymize_background_knowledge:
                self.train_df = self.anonymize_bk(self.train_df)
            if self.use_document_curation:
                self.document_curation(self.train_df, self.eval_dfs)
            self.pretreatment_done = True

        if self.save_pretreatment and self.pretreatment_done:
            self.save_pretreatment_dfs(self.train_df, self.eval_dfs)

    def read_data(self) -> pd.DataFrame:
        if self.data_file_path.endswith(".json"):
            data_df = pd.read_json(self.data_file_path)
        elif self.data_file_path.endswith(".csv"):
            data_df = pd.read_csv(self.data_file_path)
        else:
            raise Exception(f"Unrecognized file extension for data file [{self.data_file_path}]. Compatible formats are JSON and CSV.")
        
        if not self.individual_name_column in data_df.columns:
            raise Exception(f"Dataframe does not contain the individual name column {self.individual_name_column}")
        if not self.background_knowledge_column in data_df.columns:
            raise Exception(f"Dataframe does not contain the background knowledge column {self.background_knowledge_column}")
        if self.dev_set_column_name is not False and not self.dev_set_column_name in data_df.columns:
            raise Exception(f"Dataframe does not contain the dev set column {self.dev_set_column_name}")
        
        anon_cols = [col_name for col_name in data_df.columns if not col_name in [self.individual_name_column, self.background_knowledge_column]]        
        if len(anon_cols) == 0:
            raise Exception(f"Dataframe does not contain columns with texts to re-identify, only individual name and background knowledge columns")
        
        data_df.sort_values(self.individual_name_column).reset_index(drop=True, inplace=True)

        return data_df

    def split_data(self, data_df:pd.DataFrame):
        data_df.replace('', np.nan, inplace=True)   # Replace empty texts by NaN

        train_cols = [self.individual_name_column, self.background_knowledge_column]
        train_df = data_df[train_cols].dropna().reset_index(drop=True)

        eval_columns = [col for col in data_df.columns if col not in train_cols]
        eval_dfs = {col:data_df[[self.individual_name_column, col]].dropna().reset_index(drop=True) for col in eval_columns}

        return train_df, eval_dfs

    def get_individuals(self, train_df:pd.DataFrame, eval_dfs:dict):
        train_individuals = set(train_df[self.individual_name_column])
        eval_individuals = set()
        for name, eval_df in eval_dfs.items():
            if name != self.dev_set_column_name: # Exclude dev_set from these statistics
                eval_individuals.update(set(eval_df[self.individual_name_column]))
        all_individuals = train_individuals.union(eval_individuals)
        no_train_individuals = eval_individuals - train_individuals
        no_eval_individuals = train_individuals - eval_individuals

        return train_individuals, eval_individuals, all_individuals, no_train_individuals, no_eval_individuals

    def get_individuals_labels(self, all_individuals:set):
        sorted_indvidiuals = sorted(list(all_individuals)) # Sort individuals for ensuring same order every time (required for automatic loading)
        label_to_name = {idx:name for idx, name in enumerate(sorted_indvidiuals)}
        name_to_label = {name:idx for idx, name in label_to_name.items()}
        num_labels = len(name_to_label)

        return label_to_name, name_to_label, num_labels

    def show_data_stats(self, train_df:pd.DataFrame, eval_dfs:dict, no_eval_individuals:set, no_train_individuals:set, eval_individuals:set):
        logging.info(f"Number of background knowledge documents for training: {len(train_df)}")

        eval_n_dict = {name:len(df) for name, df in eval_dfs.items()}
        logging.info(f"Number of protected documents for evaluation: {eval_n_dict}")

        if len(no_eval_individuals) > 0:
            logging.info(f"No protected documents found for {len(no_eval_individuals)} individuals.")
        
        if len(no_train_individuals) > 0:
            max_risk = (1 - len(no_train_individuals) / len(eval_individuals)) * 100
            logging.info(f"No background knowledge documents found for {len(no_train_individuals)} individuals. Re-identification risk limited to {max_risk:.3f}% (excluding dev set).")

    def load_spacy_nlp(self):
        if self.spacy_nlp is None:
            self.spacy_nlp = en_core_web_lg.load()
        return self.spacy_nlp

    def anonymize_bk(self, train_df:pd.DataFrame) -> pd.DataFrame:
        spacy_nlp = self.load_spacy_nlp()        
        train_anon_df = self.anonymize_df(train_df, spacy_nlp)

        if self.only_use_anonymized_background_knowledge:
            train_df = train_anon_df # Overwrite train dataframe with the anonymized version
        else:
            train_df = pd.concat([train_df, train_anon_df], ignore_index=True, copy=False) # Concatenate to train dataframe

        return train_df

    def anonymize_df(self, df, spacy_nlp, gc_freq=5) -> pd.DataFrame:
        assert len(df.columns) == 2 # Columns expected: name and text
        anonymized_df = df.copy(deep=True)

        column_name = anonymized_df.columns[1]
        texts = anonymized_df[column_name]
        for i, text in enumerate(tqdm(texts, desc=f"Anonymizing {column_name} documents")):
            new_text = text

            doc = spacy_nlp(text) # Usage of spaCy NER (https://spacy.io/api/entityrecognizer)
            for e in reversed(doc.ents): # Reversed to not modify the offsets of other entities when substituting
                start = e.start_char
                end = start + len(e.text)
                new_text = new_text[:start] + e.label_ + new_text[end:]

            del doc
            if i % gc_freq == 0:
                gc.collect()
            texts[i] = new_text

        return anonymized_df

    def document_curation(self, train_df:pd.DataFrame, eval_dfs:dict):
        spacy_nlp = self.load_spacy_nlp()
        self.curate_df(train_df, spacy_nlp)
        for eval_df in eval_dfs.values():
            self.curate_df(eval_df, spacy_nlp)

    def curate_df(self, df, spacy_nlp, gc_freq=5):
        assert len(df.columns) == 2

        special_characters_pattern = re.compile(r"[^ \nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ./]+")
        stopwords = spacy_nlp.Defaults.stop_words

        column_name = df.columns[1]
        texts = df[column_name]
        for i, text in enumerate(tqdm(texts, desc=f"Preprocessing {column_name} documents")):
            doc = spacy_nlp(text) # Usage of spaCy (https://spacy.io/)
            new_text = ""   # Start text string
            for token in doc:
                if token.text not in stopwords:
                    token_text = token.lemma_ if token.lemma_ != "" else token.text
                    token_text = re.sub(special_characters_pattern, '', token_text)
                    new_text += ("" if token_text == "." else " ") + token_text

            del doc
            if i % gc_freq == 0:
                gc.collect()

            texts[i] = new_text

    def save_pretreatment_dfs(self, train_df:pd.DataFrame, eval_dfs:dict):
        with open(self.pretreated_data_path, "w") as f:
            f.write(json.dumps((train_df.to_json(orient="records"),
                                {name:df.to_json(orient="records") for name, df in eval_dfs.items()})))        


    def load_pretreatment(self):
        with open(self.pretreated_data_path, "r") as f:
            (train_df_json_str, eval_dfs_jsons) = json.load(f)        
        
        train_df = pd.read_json(StringIO(train_df_json_str))
        eval_dfs = OrderedDict([(name, pd.read_json(StringIO(df_json))) for name, df_json in eval_dfs_jsons.items()])

        return train_df, eval_dfs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the data preprocessor for anonymization and curation.")
    parser.add_argument("--input_file", "-i", type=str, default="data/wiki_toy.json", help="Path to the input data file.")
    parser.add_argument("--output_file", "-o", type=str, default="data/wiki_toy_done.json", help="Path to save the preprocessed data.")
    args = parser.parse_args()
    data_preprocessor = DataPreprocessor(
        input_file=args.input_file,
        output_file=args.output_file,
    )
    data_preprocessor.run(verbose=True)
    logging.info("Data preprocessing completed successfully.")

    train_df, eval_dfs = data_preprocessor.load_pretreatment()
    print(train_df.head())
    print(eval_dfs["spacy_abstract"].head())
