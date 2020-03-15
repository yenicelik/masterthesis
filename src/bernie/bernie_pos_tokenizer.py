import spacy
from transformers import BertTokenizer

from src.bernie.bernie_pos_model import BerniePoSModel
from src.resources.augment import augment_sentence_by_pos


class BerniePoSTokenizer(BertTokenizer):
    """
        Implements the BERT tokenizer,
        with the difference that we split up by PoS tokens,
        and replace these more special tokens
        run -> run_NOUN, run_VERB, ...
            where
        run_NOUN = run_1
        run_VERB = run_2
    """

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

        doc = self.nlp(sentence)

        # For all the above target words
        for token in doc:
            # Identify if it is in sentence

            # If it is not a target word, don't modify
            if token.text not in self.split_tokens:
                new_sentence.append(token.text)

            else:
                # print("Found token in split tokens!", token.text)

                pos = token.pos_
                if token.text in self.replace_dict.keys():
                    # print("Fill into existing dictionary")

                    # retrieve index of item
                    idx = self.replace_dict[token.text].index(pos) if pos in self.replace_dict[token.text] else -1
                    if idx == -1:
                        self.replace_dict[token.text].append(pos)
                        idx = self.replace_dict[token.text].index(pos)
                        assert idx >= 0

                else:
                    # print("Make a new spot")

                    self.replace_dict[token.text] = [pos, ]
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
                 **kwargs
                 ):
        print("kwargs are")
        print(kwargs)

        super(BerniePoSTokenizer, self).__init__(
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

    def inject_model(self, bernie_model: BerniePoSModel):
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
        tokens_to_add = [new_token,]
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
        new_text = self._augment_sentence_and_inject_token(text)
        # new_text = augment_sentence_by_pos(text, self.nlp, self.split_tokens, self._replace_dict)

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
    tokenizer = BerniePoSTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = BerniePoSTokenizer()

    print("Tokenizer is: ", tokenizer)

    tokenizer.inject_split_token("book")

    # Toknize an example sentence
    example_sentence = "It costs the Open Content Alliance as much as $30 to scan each book, a cost shared by the group’s members and benefactors, so there are obvious financial benefits to libraries of Google’s wide-ranging offer, started in 2004."
    tokenizer_item = tokenizer.tokenize(example_sentence)
    print(tokenizer_item)
