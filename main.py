import json
import re
from collections import Counter, defaultdict
from os.path import splitext

import nltk
import pandas as pd
import plotly.express as px
import plotly.io as pio
import spacy
from gensim import corpora
from gensim.models import LdaModel
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import (
    coleman_liau_index,
    flesch_kincaid_grade,
    flesch_reading_ease,
    smog_index,
)

nltk.download("rslp")
from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()
nlp = spacy.load("pt_core_news_lg")
nlp.max_length = 1800000


class Idioleto:
    def __init__(self, path_data, metadata):
        self.path_data = path_data
        f = open(metadata)
        self.metadata = self.__read_metadata(metadata)

    def __read_metadata(self, metadata):
        with open(metadata, "r") as f:
            data = json.load(f)
            return data

    def __remove_stopwords(self, text):
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop]
        return " ".join(tokens)

    def remove_stopwords(self):
        for elem in self.metadata:
            with open(f"data/{elem.get('titulo')}.txt", "r") as f_read:
                text_without_stopwords = self.__lemmatize_text(
                    self.__remove_stopwords(self.__remove_punctuation(f_read.read()))
                )
                with open(
                    f"data/{elem.get('titulo')}_processed.txt", "w", encoding="utf-8"
                ) as f_write:
                    f_write.write(re.sub("\s+", " ", text_without_stopwords.lower()))

    def __remove_punctuation(self, text):
        doc = nlp(text)
        text_without_punctuation = " ".join(
            [
                self.__remove_non_alphanumeric(token.text)
                for token in doc
                if not token.is_punct
            ]
        )
        return text_without_punctuation

    def __lemmatize_text(self, text):
        doc = nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        return lemmatized_text

    def __remove_non_alphanumeric(self, text):
        cleaned_string = re.sub(
            r"[^a-zA-Z0-9áàãéèíóòõúùüçÁÀÃÉÈÍÓÒÕÚÙÜÇêÊòúÔïÏüÜãõÃÕáéíóúÁÉÍÓÚ\s]",
            "",
            text,
        )
        return cleaned_string

    def generate_stemmer(self):
        for elem in self.metadata:
            with open(f"data/{elem.get('titulo')}_processed.txt", "r") as f_read:
                text = f_read.read()
                tokens = text.split()
                stemmed_text = " ".join([stemmer.stem(token) for token in tokens])
                with open(
                    f"data/{elem.get('titulo')}_stemmed.txt", "w", encoding="utf-8"
                ) as f_write:
                    f_write.write(stemmed_text)

    def count_of_terms(self, normalize=False, quantity=10):
        dict_result = {}
        for elem in self.metadata:
            with open(f"data/{elem.get('titulo')}_processed.txt", "r") as f_read:
                doc = nlp(f_read.read())
                words = [
                    token.text
                    for token in doc
                    if not token.is_stop and not token.is_punct
                ]
                word_frequencies = Counter(words)

                words_most_common = dict(word_frequencies.most_common(quantity))
                if normalize:
                    words_most_common = {
                        k: float(v) / len(words) for k, v in words_most_common.items()
                    }

                dict_result[elem.get("titulo")] = words_most_common
        with open("temp/count_of_terms.json", "w", encoding="utf-8") as outfile:
            json.dump(dict_result, outfile, ensure_ascii=False)

    def similarity(self, type_data="processed"):
        all_texts = []
        for elem in self.metadata:
            with open(f"data/{elem.get('titulo')}_{type_data}.txt", "r") as f_read:
                all_texts.append(f_read.read())

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        columns = [elem.get("titulo") for elem in self.metadata]
        fig = px.imshow(
            similarity_matrix,
            text_auto=True,
            labels=dict(x="Obras", y="Obras"),
            x=columns,
            y=columns,
            color_continuous_scale="Inferno",
        )
        fig.update_traces(
            texttemplate="%{z:.3f}",
        )
        layout = fig.layout.update(autosize=False, width=2000 * 1.2, height=2400 * 1.2)
        fig.update_xaxes(side="top")
        pio.write_image(fig, "fig_similaridade.png")
        fig.write_html("page_similaridade.html")

    def reability_metrics(self, type_data="processed"):
        readability_scores = dict()
        for elem in self.metadata:
            file_path, _ = splitext(f"data/{elem.get('arquivo')}")
            with open(f"{file_path}_{type_data}.txt", "r") as f_read:
                text = f_read.read()

                readability_scores[file_path] = {
                    "Flesch Reading Ease": flesch_reading_ease(text),
                    "SMOG Index": smog_index(text),
                    "Flesch-Kincaid Grade": flesch_kincaid_grade(text),
                    "Coleman-Liau Index": coleman_liau_index(text),
                }
        with open("temp/reability_metrics.json", "w", encoding="utf-8") as outfile:
            json.dump(readability_scores, outfile, ensure_ascii=False)

        return readability_scores

    def lexical_diversity(self, type_data="processed"):
        lexical_score = dict()
        for elem in self.metadata:
            file_path, _ = splitext(f"data/{elem.get('arquivo')}")
            with open(f"{file_path}_{type_data}.txt", "r") as f_read:
                text = f_read.read()
                fdist = FreqDist(text)
                num_unique_words = len(fdist)
                num_total_words = len(text)
                lexical_diversity = num_unique_words / num_total_words

                lexical_score[file_path] = {
                    "Number of Words": num_total_words,
                    "Number of Unique Words": num_unique_words,
                    "Lexical Diversity": lexical_diversity,
                }
        with open("temp/lexical_diversity.json", "w", encoding="utf-8") as outfile:
            json.dump(lexical_score, outfile, ensure_ascii=False)

        return lexical_score

    def return_all_texts(self, type_data="processed"):
        all_text = []
        dict_files_names = dict()
        for i, elem in enumerate(self.metadata):
            file_name = elem.get("titulo")
            dict_files_names[f"Document {i}"] = file_name
            file_path = (
                f"data/{file_name}_{type_data}.txt"
                if type_data
                else f"data/{file_name}.txt"
            )

            with open(file_path, "r") as f_read:
                text = f_read.read()
                all_text.append(text)
        return all_text, dict_files_names

    def count_sequences(
        self, minimum_length_sequence=3, max_length_sequencen=10, type_data=None
    ):
        for length_sequence in range(minimum_length_sequence, max_length_sequencen + 1):
            dict_data = {}
            for elem in self.metadata:
                file_name = elem.get("titulo")
                file_path = (
                    f"data/{file_name}_{type_data}.txt"
                    if type_data
                    else f"data/{file_name}.txt"
                )
                most_common_sequences = {}
                with open(file_path, "r") as f_read:
                    text = f_read.read()

                    words = re.findall(r"\b\w+\b", text)
                    subsequence_counts = defaultdict(int)

                    for i in range(len(words) - length_sequence):
                        subsequence = " ".join(words[i : i + length_sequence])
                        subsequence_counts[subsequence] += 1

                    filtered_sequences = {
                        seq: count
                        for seq, count in subsequence_counts.items()
                        if count > 1
                    }

                    most_common_sequences = dict(
                        sorted(
                            filtered_sequences.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    )
                dict_data[file_name] = most_common_sequences
            with open(
                f"temp/count_sequence_{length_sequence}.json", "w", encoding="utf-8"
            ) as outfile:
                json.dump(dict_data, outfile, ensure_ascii=False)

    def lda_tags(self, type_data="processed"):
        all_text, dict_files_names = self.return_all_texts(type_data=type_data)

        all_text_tokened = [text.split(" ") for text in all_text]
        dictionary = corpora.Dictionary(all_text_tokened)
        corpus = [dictionary.doc2bow(text) for text in all_text_tokened]
        lda_model = LdaModel(
            corpus, num_topics=9, id2word=dictionary, random_state=1234
        )

        idx_list = []
        topic_str_list = []
        for idx, terms in lda_model.print_topics():
            idx_list.append(idx)
            topic_str_list.append(", ".join(re.findall(r'"([^"]+)"', str(terms))))
        topic_distribution = [lda_model[doc] for doc in corpus]
        all_data = []
        for i, doc_topics in enumerate(topic_distribution):
            dict_row = {"document": dict_files_names[f"Document {i}"]}
            dict_row.update({doc_topic[0]: doc_topic[1] for doc_topic in doc_topics})
            for idx in range(9):
                if not idx in dict_row:
                    dict_row[idx] = 0

            all_data.append(dict_row)

        df = pd.DataFrame(all_data)
        fig = px.imshow(
            df[idx_list],
            text_auto=True,
            labels=dict(x="Topic", y="Document", color="Rating"),
            x=topic_str_list,
            y=df["document"].tolist(),
        )

        layout = fig.layout.update(autosize=False, width=1600, height=2400)
        fig.update_xaxes(side="top")
        pio.write_image(fig, "fig_lda.png")

        fig.write_html("page_lda.html")


if __name__ == "__main__":
    idioleto = Idioleto("data", "data/metadata.json")
    # idioleto.remove_stopwords()
    # idioleto.generate_stemmer()
    # idioleto.count_of_terms(normalize=False, quantity=50)
    # idioleto.reability_metrics()
    # idioleto.lexical_diversity()
    # idioleto.lda_process(type_data=None)
    # idioleto.similarity(type_data="processed")
    # idioleto.lda_tags(type_data="processed")
    idioleto.count_sequences()
