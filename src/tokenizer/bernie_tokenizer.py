import spacy
from transformers import BertTokenizer


class CustomTokenizer(BertTokenizer):
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

        super(CustomTokenizer, self).__init__(
            self,
            vocab_file,
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

        self.injected_new_tokens = False

        # This should probably
        self.replace_dict = dict()
        self.nlp = spacy.load("en_core_web_sm")

    def _tokenize(self, text):
        assert self.injected_new_tokens, ("Must inject new tokens before you can use the Bernie tokenizer!")

        # Apply the nlp tokenization, replace tokens,
        # retokenize and pass this into the BERT Tokenizer function


if __name__ == "__main__":
    print("Run the tokenizer on an example sentence!")
