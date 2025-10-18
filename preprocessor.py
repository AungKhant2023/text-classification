# preprocessor.py

import re
import os

class MyanmarTextPreprocessor:
    def __init__(self, dict_path: str, stop_path: str = None):
        # Load dictionary
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        self.dictionary = self.load_dictionary(dict_path)

        # Load stopwords if path provided, else empty set
        if stop_path and os.path.exists(stop_path):
            self.stopwords = self.load_stopwords(stop_path)
        else:
            self.stopwords = set()

    def load_dictionary(self, dict_path):
        dictionary = set()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    dictionary.add(word)
        return dictionary

    def load_stopwords(self, stopword_path):
        stopwords = set()
        with open(stopword_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        return stopwords

    def merge_with_dictionary(self, syllables):
        merged_tokens = []
        i = 0
        while i < len(syllables):
            matched = False
            for j in range(len(syllables), i, -1):
                combined = ''.join(syllables[i:j])
                if combined in self.dictionary:
                    merged_tokens.append(combined)
                    i = j
                    matched = True
                    break
            if not matched:
                merged_tokens.append(syllables[i])
                i += 1
        return merged_tokens

    def preprocessing(self, text: str):
        # Myanmar + English + number tokenization
        text = re.sub(
            r"(([A-Za-z0-9]+)|[က-အ|ဥ|ဦ](င်္|[က-အ][ှ]*[့း]*[်]|္[က-အ]|[ါ-ှႏꩻ][ꩻ]*){0,}|.)",
            r"\1 ",
            text
        )
        text = text.strip().split()
        merged_tokens = self.merge_with_dictionary(text)
        filtered_tokens = [token for token in merged_tokens if token not in self.stopwords]
        return ' '.join(filtered_tokens)











# import re

# class MyanmarTextPreprocessor():
#     def __init__(self, dict_path: str, stop_path: str):
#         self.dictionary = self.load_dictionary(dict_path)
#         self.stopwords = self.load_stopwords(stop_path)

#     def load_dictionary(self, dict_path):
#         dictionary = set()
#         with open(dict_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 word = line.strip()
#                 if word:
#                     dictionary.add(word)
#         return dictionary

#     def load_stopwords(self, stopword_path):
#         stopwords = set()
#         with open(stopword_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 word = line.strip()
#                 if word:
#                     stopwords.add(word)
#         return stopwords

#     def merge_with_dictionary(self, syllables):
#         merged_tokens = []
#         i = 0
#         while i < len(syllables):
#             matched = False
#             for j in range(len(syllables), i, -1):
#                 combined = ''.join(syllables[i:j])
#                 if combined in self.dictionary:
#                     merged_tokens.append(combined)
#                     i = j
#                     matched = True
#                     break
#             if not matched:
#                 merged_tokens.append(syllables[i])
#                 i += 1
#         return merged_tokens

#     def preprocessing(self, text: str):
#         text = re.sub(r"(([A-Za-z0-9]+)|[က-အ|ဥ|ဦ](င်္|[က-အ][ှ]*[့း]*[်]|္[က-အ]|[ါ-ှႏꩻ][ꩻ]*){0,}|.)", r"\1 ", text)
#         text = text.strip().split()
#         merged_tokens = self.merge_with_dictionary(text)
#         filtered_tokens = [token for token in merged_tokens if token not in self.stopwords]
#         return ' '.join(filtered_tokens)


    # def preprocessing(self, text: str):
    # # Regex to capture Myanmar syllables + English words + numbers
    #     pattern = re.compile(
    #         r"[က-အ](?:[\u103B-\u103E\u103A\u102B-\u103E\u1037\u1038\u1039]*)"  # Myanmar syllable
    #         r"|[a-zA-Z0-9]+",  # English/Numbers
    #         re.UNICODE
    #     )

    #     text = pattern.findall(text)

    #     # Merge tokens if they exist in dictionary
    #     merged_tokens = self.merge_with_dictionary(text)

    #     # Remove stopwords
    #     filtered_tokens = [token for token in merged_tokens if token not in self.stopwords]

    #     return ' '.join(filtered_tokens)

