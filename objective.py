import re
import nltk
import numpy as np
import torch
import random
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class ObjectiveTest:
    def __init__(self, data, noOfQues):
        self.summary = data
        self.noOfQues = noOfQues
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        trivial_sentences = []
        for sent in sentences:
            trivial = self.identify_trivial_sentences(sent)
            if trivial:
                trivial_sentences.append(trivial)
        return trivial_sentences

    def identify_trivial_sentences(self, sentence):
        tokens = nltk.word_tokenize(sentence)  # Tokenize the sentence into words
        tags = nltk.pos_tag(tokens)

        if tags[0][1] == "RB" or len(tokens) < 4:
            return None

        noun_phrases = list()
        grammar = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
                   {<NN>+<IN|DT>*<NNP>+}
                   {<NNP>+<NNS>*}
        """
        chunker = nltk.RegexpParser(grammar)
        pos_tokens = nltk.tag.pos_tag(tokens)
        tree = chunker.parse(pos_tokens)

        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                temp = " ".join(word for word, _ in subtree)
                noun_phrases.append(temp.strip())

        # If there are no noun phrases, return None
        if not noun_phrases:
            return None

        # Randomly select one noun phrase to replace with a blank
        selected_phrase = random.choice(noun_phrases)

        # Create a blank phrase
        blanks_phrase = "__________"
        expression = re.compile(re.escape(selected_phrase), re.IGNORECASE)

        # Replace only the first occurrence of the selected phrase with blanks
        sentence_with_blank = expression.sub(blanks_phrase, sentence, count=1)

        trivial = {
            "Answer": selected_phrase,
            "Question": sentence_with_blank,
            "Similar": self.answer_options(selected_phrase)
        }

        return trivial

    def answer_options(self, word):
        # Get word embeddings for the word using BERT
        inputs = self.tokenizer(word, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the embeddings for the word
        word_embedding = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling

        # Use WordNet to find candidates
        candidates = wn.synsets(word, pos="n")
        if not candidates:
            return []
        
        similar_words = []
        for candidate in candidates:
            for lemma in candidate.lemmas():
                candidate_word = lemma.name().replace("_", " ")
                if candidate_word != word:
                    # Get embeddings for candidate words
                    candidate_inputs = self.tokenizer(candidate_word, return_tensors='pt')
                    with torch.no_grad():
                        candidate_outputs = self.model(**candidate_inputs)
                    
                    candidate_embedding = candidate_outputs.last_hidden_state.mean(dim=1)

                    # Calculate cosine similarity
                    similarity = cosine_similarity(word_embedding.numpy(), candidate_embedding.numpy())
                    if similarity > 0.5:  # Adjust threshold as necessary
                        similar_words.append(candidate_word)

                    if len(similar_words) >= 8:
                        return similar_words

        return similar_words

    def generate_test(self):
        trivial_pair = self.get_trivial_sentences()
        question_answer = []

        # Adjust the check based on your actual data structure
        for que_ans_dict in trivial_pair:
            # Ensure we're checking the correct keys
            if que_ans_dict.get("Answer") and que_ans_dict.get("Question"):
                question_answer.append(que_ans_dict)

        if len(question_answer) == 0:
            raise ValueError("No valid questions available to generate.")

        question = []
        answer = []
        num_questions_to_generate = min(int(self.noOfQues), len(question_answer))

        while len(question) < num_questions_to_generate:
            rand_num = np.random.randint(0, len(question_answer))
            selected_question = question_answer[rand_num]["Question"]

            if selected_question not in question:
                question.append(selected_question)
                answer.append(question_answer[rand_num]["Answer"])

        return question, answer

