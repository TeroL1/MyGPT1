import dill

class BPE():
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def fit(self, text: str, verbose: int = None) -> None:
        unique = __class__._findUnique(text)
        tokenized_text = __class__._findBaseTokens(text)

        while len(unique) < self.vocab_size:
            most_frequent = __class__._findMostFrequent(tokenized_text)
            unique.append(most_frequent)
            tokenized_text = __class__._rebuildTokenizedText(tokenized_text, most_frequent)

            if verbose and (len(unique) % verbose == 0 or len(unique) == self.vocab_size):
              print(f"{len(unique)} / {self.vocab_size}")
        self._buildId2Token(unique)
        self._buildToken2Id(unique)

    def encode(self, text: str) -> list:
        n = len(text)

        new_tokenized_text= []

        index = 0
        while index < n:
            new_token, index = self._encodeTokenStep(text, index)
            new_token = self.token2id[new_token]
            new_tokenized_text.append(new_token)

        return new_tokenized_text

    def decode(self, token_ids: list) -> list:
        n = len(token_ids)
        tokens = []

        for index in token_ids:
            tokens.append(self.id2token[index])

        text = ''.join(tokens)

        return text

    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)
        print(f"Объект сохранён в {filename}")


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = dill.load(f)

        print(f"Объект загружен из {filename}")
        return obj

    def _buildId2Token(self, unique: list) -> None:
        self.id2token = dict()

        for index, value in enumerate(unique):
            self.id2token[index] = value

    def _buildToken2Id(self, unique: list) -> None:
        self.token2id = dict()

        for value, index in enumerate(unique):
            self.token2id[index] = value

    def _encodeTokenStep(self, text: str, index: int) -> tuple:
        candidates = [token for token in self.token2id if text.startswith(token, index)]

        if not candidates:
            return text[index], index + 1

        token = max(candidates, key = len)
        i = index + len(token)

        return token, i

    def _findCandidatesToken(self, token: str) -> set:
        candidates = set()

        for key in self.token2id.keys():
            if token in key:
                candidates.add(key)

        return candidates

    @staticmethod
    def _findUnique(text: str) -> list:
        unique = list(set(text))
        unique.sort()

        return unique

    @staticmethod
    def _findBaseTokens(text: str) -> list:
        unique = list(text)

        return unique

    @staticmethod
    def _findMostFrequent(tokenized_text: list) -> str:
        n = len(tokenized_text)
        frequency = dict()
        first_appearance = dict()

        for index in range(n - 1):
            current = tokenized_text[index] + tokenized_text[index + 1]
            frequency[current] = frequency.get(current, 0) + 1

            if current not in first_appearance:
                first_appearance[current] = index

        most_frequent = max(frequency.items(), key = lambda x: (x[1], -first_appearance[x[0]]))[0]

        return most_frequent

    @staticmethod
    def _rebuildTokenizedText(tokenized_text: list, most_frequent: str) -> list:
        n = len(tokenized_text)
        new_tokenized_text = []

        index = 0
        while index < n - 1:
            current = tokenized_text[index] + tokenized_text[index + 1]
            if current == most_frequent:
                new_tokenized_text.append(current)
                index += 2

            else:
                new_tokenized_text.append(tokenized_text[index])
                index += 1
        if index == n - 1:
            new_tokenized_text.append(tokenized_text[index])

        return new_tokenized_text

    @staticmethod
    def _findCandidatesMask(candidates, mask) -> set:
        new_candidates = set()

        for candidate in candidates:
            if candidate.startswith(mask):
                new_candidates.add(candidate)

        return new_candidates