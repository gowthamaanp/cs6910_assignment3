from ..utils.config import *  # Import necessary constants from the config file

class CharSet:
    def __init__(self, language):
        self.lanugage = language  # Store the input language
        self.char2index = {'^': SOS_TOKEN, '$': EOS_TOKEN, '#': PAD_TOKEN}  # Initialize a dictionary to map characters to numeric indices
        self.index2char = {SOS_TOKEN: '^', EOS_TOKEN: '$', PAD_TOKEN: '#'}  # Initialize a dictionary to map indices to characters
        self._unicode_range = {  # Define the Unicode range for different languages
            'eng': (97, 122),    # English (lowercase)
            'tam': (2944, 3071), # Tamil
            'hin': (2304, 2431)  # Hindi
        }
        self._build_charset()  # Call the method to build the character set

    def _build_charset(self):
        unicode_range = self._unicode_range[self.lanugage]  # Get the Unicode range for the specified language
        chars = [chr(char) for char in range(unicode_range[0], unicode_range[1]+1)]  # Generate a list of characters based on the Unicode range
        for index, char in enumerate(chars):
            idx = index + 3  # Start indexing from 3 (after the special tokens)
            self.char2index[char] = idx  # Add the character to the char2index dictionary with its index
            self.index2char[idx] = char  # Add the index to the index2char dictionary with its character

    def get_length(self):
        return len(self.char2index)  # Return the length of the character set