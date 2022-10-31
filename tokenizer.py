import re
import string


class Tokenizer:
    # define punctuation chars !"#$%&'()*+,./:;<=>?@[\]^_`{|}~  plus some unicode punct ¿؟¡¢
    expanded_punct = string.punctuation + chr(191) + chr(1567) + chr(161) + chr(162)
    # £¥©®§
    expanded_punct = expanded_punct + chr(163) + chr(165) + chr(169) + chr(174) + chr(167)
    #  unicode additions (found with https://unicode-table.com/en/blocks/general-punctuation/)
    #     this picks up en-dash, em-dash and punctuation seen in (at least) Russian
    for i in range(8192, 8303):
        if i == 8293:  # unused
            continue
        expanded_punct = expanded_punct + chr(i)

    spaces = ""
    for i in range(0, len(expanded_punct)):
        spaces += " "

    ttable = str.maketrans(expanded_punct, spaces)

    def __init__(self):
        pass

    def extract_tokens(self, str_content):
        """
        Clean given string and return the cleaned list of tokens.
        Punctuation, excess whitespace, control chars, funny stuff, is removed after being used
        to split into tokens.
        :param str_content: raw string
        :return: token list, preserving order from document.
        """
        # convert control and EOL chars to space
        str_content = re.sub(r'[\x00-\x1F]+', ' ', str_content)
        str_content = str_content.translate(self.ttable)
        # split by whitespace
        tokens = list(filter(None, str_content.split()))

        tokens = [w.translate(self.ttable) for w in tokens]
        return tokens
