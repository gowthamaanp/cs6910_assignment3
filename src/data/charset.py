class CharSet:
    def __init__(self, name):
        self.name = name
        self.char2index = {'#': 0, '$': 1}
        self.index2char = {0: '#', 1: '$'}
        self._unicode_range = {
            'eng': (97, 122),
            'tam': (2944, 3071),
            'hin': (2304, 2431)
        }
        self._build_charset()
    
    def _build_charset(self):
        unicode_range = self._unicode_range[self.name]
        chars = [chr(char) for char in range(unicode_range[0], unicode_range[1]+1)]
        for index, char in enumerate(chars):
            idx = index+2
            self.char2index[char] = idx
            self.index2char[idx] = char