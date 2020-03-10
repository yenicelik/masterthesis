import spacy
from transformers import BertTokenizer

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

        self.split_tokens = []

        # This should probably
        # The dictionary which records which index corresponds to which meaning
        self._replace_dict = dict()
        self.nlp = spacy.load("en_core_web_sm")

        # what are the target words

    @property
    def replace_dict(self):
        return self._replace_dict

    # TODO: What about a function one has to call each time before a sentence is input,
    # that both updates the tokenizer and the model, if the token is not present in the model

    def inject_split_token(self, split_word, n=5):
        """

        # Do the part where you add the few additional tokens to the vocabulary ...
        # Need to do this dynamically, as we don't know all tokens beforehand

        :param target_words: A word which will be replaced by a PoS specialization
        :return:
        """
        assert len(split_word) > 0, ("Split word is empty", split_word)
        self.split_tokens = split_word

        old_vocab_size = len(self.added_tokens_decoder)

        # Check if the word is already in the vocabulary. If not, we cannot allow this split
        # (for convenience, also a rare case)
        if split_word not in self.vocab:
            print(f"'{split_word}' cannot be added, as it is not part of the standard vocabulary")
            return -1, 0

        # 0. Find the target word idx in the vocabulary
        word_idx = self.vocab[split_word]

        # 1. Expand the tokenizer by this words ...
        tokens_to_add = [(f'{split_word}_{i}') for i in range(n)]
        number_new_tokens = len(tokens_to_add)
        added_tokens = self.add_tokens(tokens_to_add)

        # 2. Check if the new dimensions conform
        assert added_tokens == n
        new_vocab_size = len(tokenizer.added_tokens_decoder)
        assert new_vocab_size == old_vocab_size + n, (new_vocab_size, old_vocab_size, n)

        # 3. Test if adding to the tokenizer was successful, by checking if this token converts to any integer id
        assert all([self.convert_tokens_to_ids(x) for x in tokens_to_add])

        # 4. The idx of this word is returned, such that any the copy-over policy for the BERT model can be applied
        return word_idx, number_new_tokens

    def pre_tokenizer(self):
        """
            Checks if there are any items in the sentence which need to be added to the tokenizer.
            If this is the case, returns the original idx, and how many additional tokens need to be added for the BertModel to be registered!
        :return:
        """

        # TODO: Make a policy, s.t. when the number of additional vectors are full, you cannot add more and it defaults to run_0
        # This happens during live-tokenization!

        # TODO: What about the case where we don't realize the case of run
        raise NotImplementedError

    def tokenize(self, text, **kwargs):
        assert self.split_tokens, ("Must inject new tokens before you can use the Bernie tokenizer!")
        assert self.split_tokens, ("Injection of new tokens must bee specified")

        print("Previous text is: ", text)
        # Apply the nlp tokenization, replace tokens,
        # retokenize and pass this into the BERT Tokenizer function
        new_text = augment_sentence_by_pos(text, self.nlp, self.split_tokens, self._replace_dict)

        print("New text is: ", new_text)

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
