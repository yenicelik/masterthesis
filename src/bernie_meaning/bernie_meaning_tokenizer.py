import spacy
from nltk import PorterStemmer
from transformers import BertTokenizer
import torch

from src.bernie_meaning.bernie_meaning_model import BernieMeaningModel
from src.bernie_meaning.cluster_model import predict_meaning_cluster
from src.functional.string_searchers import find_all_indecies_subarray
from src.language_models.model_wrappers.bert_wrapper import BertWrapper


class BernieMeaningTokenizer(BertTokenizer):
    """
        Implements the BERT tokenizer,
        with the difference that we split up by PoS tokens,
        and replace these more special tokens
        run -> run_NOUN, run_VERB, ...
            where
        run_NOUN = run_1
        run_VERB = run_2
    """

    def _get_bert_embedding_for_word(self, word: str, sentence: str):
        """
            Retrieves a single embedding for the word
        :param word:
        :param sentence:
        :return:
        """
        assert isinstance(word, str), ("Word should be a string!", word, type(word))
        assert isinstance(sentence, str), ("Sentence should be a string!", type(sentence), sentence)

        # strip word if further needed maybe?
        tokenized_word = self.bert_embedding_retriever_model.tokenizer.tokenize(word)
        tokenized_word_window = len(tokenized_word)  # This targets words which make up more than a single token

        # Need to tokenize the sentence as well
        # print("Sentence before tokenization is: ", sentence)
        tokenized_sentence = self.bert_embedding_retriever_model.tokenizer.tokenize(sentence)
        # print("Sentence after tokenization is: ", sentence)
        indexed_tokens = self.bert_embedding_retriever_model.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        # print("Indexed tokens are: ", indexed_tokens)
        # Get location of indexed tokens
        # print("Inputs are: ", tokenized_word, tokenized_sentence)
        # print(len(tokenized_word))
        # print("Inputs are: ", word, sentence)
        # print("Indexed tokens are", indexed_tokens)
        # tmp = find_all_indecies_subarray(tokenized_word, tokenized_sentence, corpus=None)
        # print("Tmp is: ", tmp)
        # TODO Might have to introduce a basic stemmer ...

        # print("Tokens are", tokenized_word)
        # print("Sentence tokens are", tokenized_sentence)

        tokenized_word_idx = find_all_indecies_subarray(
            subarray=tokenized_word,
            array=tokenized_sentence,
            fn_stem=self.stemmer.stem if self.stemmer is not None else None
        )[0]

        # Convert to pytorch tensors
        # Now convert to pytorch tensors..
        segments_ids = [0, ] * len(tokenized_sentence)
        # print("Segment ids are", segments_ids)
        # print("Sentence is", segments_ids)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # print("Tokens tensor and segments tensor is: ")
        # The first one should not be a single entity ...
        # print(tokens_tensor)
        # print(segments_tensors)

        outputs = self.bert_embedding_retriever_model.forward(
            tokens_tensor=tokens_tensor,
            segments_tensors=segments_tensors
        )
        word_embedding = outputs[0, tokenized_word_idx:tokenized_word_idx + tokenized_word_window, :]

        # Reshape amongst new dimension
        # And apply aggregation if desired

        assert word_embedding.shape == (1, 768), (word_embedding.shape, (1, 768))

        # Perhaps return a dictionary instead ...
        return word_embedding


    def _augment_sentence_and_inject_token(self, sentence):
        """
            Apply this function before feeding a sentence in to BERET.
            Due to the nature of the `replace_dict`,
            this function must be applied to the entire corpus before the application.

            This function takes 0.01 (0.05 completely non-cached) seconds per run, which is considered fast enough for actual runs.

            replace_dict must be a reference!!!

        :param sentence: the sentence to be replaced
        :param nlp: the spacy nlp tokenizer
        :param tgt_words: The target words which shall all be replaced
        :param replace_dict: The dictionary which translates the "meaning" to the number # MUST BE A DICT-reference!!!
        :return:
        """

        new_sentence = []

        # Here, apply the meaning-tokenization by wordnet senses

        # TODO: Do we need to do this now..? Can we use a different tokenizer, perhaps a simpler one?
        doc = self.nlp(sentence)

        # For all the above target words
        for token in doc:
            # Identify if it is in sentence

            # If it is not a target word, don't modify
            if token.text not in self.split_tokens:
                # print("Normal append")
                new_sentence.append(token.text)

            else:
                # print("Special append (found in split tokens)!", token.text)

                # TODO: Rewrite this function based on some clustering pickle files ....
                # TODO: Perhaps also do a "just-in-time" training with the BERT model -> These files can be cached!!!
                # (saving and loading will be a bit tough, but should be good)

                # Do a sentence forward pass through the vanilla BERT model
                # We use the vanilla BERT model, because this is what we cluster by ...
                # print("Input to embeddings are: ")
                # print(token.text)
                # print(sentence)
                # print()
                embedding = self._get_bert_embedding_for_word(
                    word=token.text,
                    sentence=sentence
                )

                # Check if the embedding dimensions match with whatever we want to have

                # Predict the meaning that this vector entails
                # print("Embeddings shape are: ", embedding.shape)
                # TODO: Move clustermodel savedir to args!
                context_id = predict_meaning_cluster(
                    word=f" {token.text} ", #  TODO: Must add space-padding to sample words!
                    embedding=embedding,
                    clustermodel_savedir=self.output_meaning_dir,  # '"/Users/david/GoogleDrive/_MasterThesis/savedir/cluster_model_caches",
                    knn_n=10
                )
                # Prepend context_id by "C{context_id}" s.t. it becomes a string
                context_id = f"C{context_id}"
                # print("Context id is: ", context_id)

                if token.text in self.replace_dict.keys():
                    # print("Fill into existing dictionary")

                    # retrieve index of item
                    idx = self.replace_dict[token.text].index(context_id) if context_id in self.replace_dict[token.text] else -1
                    if idx == -1:
                        # print("Specialization not found")
                        self.replace_dict[token.text].append(context_id)
                        idx = self.replace_dict[token.text].index(context_id)
                        assert idx >= 0

                else:
                    # print("Make a new spot", context_id)
                    self.replace_dict[token.text] = [context_id, ]
                    idx = 0

                # print("This is the new token...")
                # print("Replacing with ", token.text, pos)

                new_token = f"{token.text}_{idx}"

                # TODO: Put the new token to the replace-dict
                if new_token not in self.added_tokens:
                    print("Injecting new token ...", new_token, self.added_tokens)
                    # TODO: Expand tokenizer here to include the new token if not existent!
                    self.inject_split_token(split_word=token.text, new_token=new_token)
                    # Finally add it to the added tokens
                    # TODO: Expand model here if not existent! -> This is done within the above function! (hopefully, lol)

                # replace the token with the new token
                new_sentence.append(new_token)

                # remove old vocabulary

        res_str = " ".join(new_sentence)
        new_sentence = res_str \
            .replace(" .", ".") \
            .replace(" ,", ",") \
            .replace(" ’", "’") \
            .replace(" - ", "-") \
            .replace("$ ", "$")

        return new_sentence

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 tokenize_chinese_chars=True,
                 output_meaning_dir=None,
                 **kwargs
                 ):
        print("kwargs are")
        print(kwargs)

        super(BernieMeaningTokenizer, self).__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            **kwargs
        )

        self.split_tokens = set()

        # This should probably
        # The dictionary which records which index corresponds to which meaning
        self._replace_dict = dict()
        self.nlp = spacy.load("en_core_web_sm")

        # what are the target words
        self.bernie_model = None
        self.output_meaning_dir = output_meaning_dir

        self.bert_embedding_retriever_model = BertWrapper()

        self.stemmer = None  # PorterStemmer()

    @property
    def added_tokens(self):
        return set(self.added_tokens_decoder.values())

    @property
    def replace_dict(self):
        return self._replace_dict

    def set_split_tokens(self, split_tokens):
        print("Setting split tokens..")
        _split_tokens = [x.strip() for x in split_tokens]
        self.split_tokens = set(_split_tokens)

    # TODO: What about a function one has to call each time before a sentence is input,
    # that both updates the tokenizer and the model, if the token is not present in the model

    def inject_model(self, bernie_model: BernieMeaningModel):
        """
            Injecting a reference to the BernieModel.
            This allows for just-in-time insertion of new tokens that are not within the replace-dict!
        :return:
        """
        self.bernie_model = bernie_model

    def inject_split_token(self, split_word, new_token):
        """

        # Do the part where you add the few additional tokens to the vocabulary ...
        # Need to do this dynamically, as we don't know all tokens beforehand

        :param target_words: A word which will be replaced by a PoS specialization
        :return:
        """

        assert self.bernie_model is not None, (
            "BernieModel must be injected before this dynamic tokenizer can be used!")
        assert len(split_word) > 0, ("Split word is empty", split_word)

        old_additional_vocab_size = len(self.added_tokens_decoder)

        # 0. Find the target word idx in the vocabulary
        token_idx = self.vocab[split_word]

        # TODO: Make the replace-dict decide what to add and how many ...

        # 1. Expand the tokenizer by this words ...
        tokens_to_add = [new_token, ]
        number_new_additional_tokens = len(tokens_to_add)
        print("Tokens to add is: ", tokens_to_add)
        added_tokens = self.add_tokens(tokens_to_add)

        # 2. Check if the new dimensions conform
        assert added_tokens == 1
        new_additional_vocab_size = len(self.added_tokens_decoder)
        assert new_additional_vocab_size == old_additional_vocab_size + 1, (
            new_additional_vocab_size, old_additional_vocab_size, 1)

        # 3. Test if adding to the tokenizer was successful, by checking if this token converts to any integer id
        assert all([self.convert_tokens_to_ids(x) for x in tokens_to_add])

        new_vocab_size = self.vocab_size + len(self.added_tokens_decoder)

        print("New vocab size is: ", split_word, new_vocab_size, old_additional_vocab_size, new_additional_vocab_size)

        # Automatically add new tokens to the injected model
        self._expand_injected_bernie_model(new_vocab_size, token_idx, number_new_additional_tokens)

        # TODO: Make a policy, s.t. when the number of additional vectors are full, you cannot add more and it defaults to run_0
        # This happens during live-tokenization!

        # TODO: What about the case where we don't realize the case of run

        # 4. The idx of this word is returned, such that any the copy-over policy for the BERT model can be applied
        return new_vocab_size, token_idx, number_new_additional_tokens

    def _expand_injected_bernie_model(self, new_vocab_size, token_idx, number_new_additional_tokens):
        """
                Checks if there are any items in the sentence which need to be added to the tokenizer.
                If this is the case, returns the original idx, and how many additional tokens need to be added for the BertModel to be registered!
            :return:
            """

        if number_new_additional_tokens == 0:
            print(f"Word {word} was skipped!")
            return

        self.bernie_model.bert.inject_split_token(
            new_total_vocabulary_size=new_vocab_size,
            token_idx=token_idx,
            number_new_tokens=number_new_additional_tokens
        )

        assert new_vocab_size == self.bernie_model.bert.embeddings.word_embeddings.weight.shape[0], (
        new_vocab_size, self.bernie_model.bert.embeddings.word_embeddings.weight.shape)

    def tokenize(self, text, **kwargs):
        # assert self.split_tokens, ("Must inject new tokens before you can use the Bernie tokenizer!")
        # assert self.split_tokens, ("Injection of new tokens must bee specified")

        # print("Split words are")
        # print(sorted(self.split_tokens))
        # print("Previous text is: ", text)

        # Apply the nlp tokenization, replace tokens,
        # retokenize and pass this into the BERT Tokenizer function
        # print("Input text is: ", text)
        new_text = self._augment_sentence_and_inject_token(text)

        # TODO: If the new_text includes a token which is not in the vocabulary yet, include this into the vocabulary

        # print("At this point, we should have added 'book_0' to the new vocabulary ..")
        # print(self.added_tokens)

        # TODO: If some tokens _0 etc. are not in replace-dict, (from within the augment_sentence_by_pos),
        # Then add them to replace-dict, add them to tokenizer, and expand model by one ...

        # print("New text is: ", new_text)
        # print("New replace dict is: ", self.replace_dict)

        # TODO: The re-used BERT tokenizer does not use the newly discovered / generated vocabulary

        # now run the actual tokenizer
        return super().tokenize(new_text)


if __name__ == "__main__":
    print("Run the tokenizer on an example sentence!")

    # Load from pre-trained
    tokenizer = BernieMeaningTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = BerniePoSTokenizer()

    print("Tokenizer is: ", tokenizer)

    tokenizer.inject_split_token("book")

    # Toknize an example sentence
    example_sentence = "It costs the Open Content Alliance as much as $30 to scan each book, a cost shared by the group’s members and benefactors, so there are obvious financial benefits to libraries of Google’s wide-ranging offer, started in 2004."
    tokenizer_item = tokenizer.tokenize(example_sentence)
    print(tokenizer_item)
