from enum import Enum
from itertools import chain
from collections import defaultdict

import numpy as np

# phoneme types
vowel = ["a", "e", "i", "I\\", "o", "u"]
stop = list(chain.from_iterable((sound, sound + "'") for sound in "pbtdkgmn"))
continuous = list(
    chain.from_iterable(
        (sound, sound + "'") for sound in "f|v|l|s|z|r|t_s|x".split("|")
    )
)
continuous.extend(["s`", "t_s\\", "z`", "j"])

# phoneme manners
high = ["I\\", "i", "u"]
low = ["e", "a", "o"]
nasal = list(chain.from_iterable((sound, sound + "'") for sound in "mn"))
liquid_semi = list(
    chain.from_iterable((sound, sound + "'") for sound in "l|t_s|r|x".split("|"))
)
liquid_semi.append("j")
fricative = list(chain.from_iterable((sound, sound + "'") for sound in "fvsz"))
fricative.extend(["s`", "t_s\\", "z`"])
oral = list(chain.from_iterable((sound, sound + "'") for sound in "pbtdkg"))

# phoneme positions
front = list(chain.from_iterable((sound, sound + "'") for sound in "pbmfvl"))
front.extend(["I\\", "i", "e"])
middle = list(
    chain.from_iterable((sound, sound + "'") for sound in "t|d|n|s|z|t_s|r".split("|"))
)
middle.append("a")
back = list(chain.from_iterable((sound, sound + "'") for sound in "kgx"))
back.extend(["s`", "t_s\\", "z`", "j", "u", "o"])

# phoneme voicedness
unvoiced = list(
    chain.from_iterable((sound, sound + "'") for sound in "p|f|t|s|t_s|k|x".split("|"))
)
unvoiced.extend(["s`", "t_s\\"])
voiced = list(chain.from_iterable((sound, sound + "'") for sound in "bmvldnzrg"))
voiced.extend(["z`", "j", "a", "I\\", "i", "e", "o", "u"])

# phoneme palatalization
hard = [
    sound
    for sound in "p|f|b|m|v|l|I\\|t|s|t_s|d|n|z|r|a|k|s`|x|g|z`|u|o|e|i".split("|")
]
soft = [sound + "'" for sound in "pfbmvltsdnzrkgx"]
soft.extend(["t_s\\", "j", "t_s'"])


class PhonemeType(Enum):
    Stop = 0
    Continuous = 1
    Vowel = 2

    def from_xsampa(phoneme):
        if phoneme in vowel:
            return PhonemeType.Vowel

        if phoneme in stop:
            return PhonemeType.Stop

        if phoneme in continuous:
            return PhonemeType.Continuous

        return None


class PhonemeManner(Enum):
    OralFricativeHigh = 0
    NasalLiquidSemiLow = 1

    def from_xsampa(phoneme):
        if phoneme in high or phoneme in fricative or phoneme in oral:
            return PhonemeManner.OralFricativeHigh
        
        if phoneme in low or phoneme in nasal or phoneme in liquid_semi:
            return PhonemeManner.NasalLiquidSemiLow

        return None


class PhonemePosition(Enum):
    Front = 0
    Middle = 1
    Back = 2

    def from_xsampa(phoneme):
        if phoneme in front:
            return PhonemePosition.Front

        if phoneme in middle:
            return PhonemePosition.Middle

        if phoneme in back:
            return PhonemePosition.Back

        return None


class PhonemeVoice(Enum):
    Unvoiced = 0
    Voiced = 1

    def from_xsampa(phoneme):
        if phoneme in unvoiced:
            return PhonemeVoice.Unvoiced

        if phoneme in voiced:
            return PhonemeVoice.Voiced

        return None


class PhonemePalatalization(Enum):
    Hard = 0
    Soft = 1

    def from_xsampa(phoneme):
        if phoneme in hard:
            return PhonemePalatalization.Hard

        if phoneme in soft:
            return PhonemePalatalization.Soft

        return None


class WordBoundary(Enum):
    Phone = 0
    Boundary = 1


class SegmentFeatures:
    phontype = None
    manner = None
    position = None
    voice = None
    palatal = None
    boundary = WordBoundary.Phone

    def __init__(self, segment):
        if segment == "#":
            self.boundary = WordBoundary.Boundary
        else:
            self.phontype = PhonemeType.from_xsampa(segment)
            self.manner = PhonemeManner.from_xsampa(segment)
            self.position = PhonemePosition.from_xsampa(segment)
            self.voice = PhonemeVoice.from_xsampa(segment)
            self.palatal = PhonemePalatalization.from_xsampa(segment)

    def __iter__(self):
        if self.boundary == WordBoundary.Boundary:
            yield self.boundary
        else:
            yield self.phontype
            yield self.manner
            yield self.position
            yield self.voice
            yield self.palatal

    def __str__(self):
        return f"Type: {self.phontype}\nManner: {self.manner}\nPosition: {self.position}\nVoice: {self.voice}\nPalatalization: {self.palatal}\nBoundary: {self.boundary}"


class WickelPhone:
    def __init__(self, first: str, second: str, third: str):
        self.phones = [first, second, third]

    def __str__(self):
        return "{" + self.phones[0] + "}" + self.phones[1] + "{" + self.phones[2] + "}"

    def activating_features(self):
        pre, central, post = [np.array(list(SegmentFeatures(p))) for p in self.phones]

        dimensions = 5

        cols = np.array(
            [
                np.tile(pre, (dimensions * dimensions) // len(pre)),
                np.repeat(central, dimensions),
                np.tile(post, (dimensions * dimensions) // len(post)),
            ]
        )

        return np.transpose(cols)


def tokenize_xsampa(xsampa: str) -> list[str]:
    result = []
    index = 0
    length = len(xsampa)
    while index < length:
        # search for joint consonants t_s, t_s', t_s\
        if length - index > 1:
            if xsampa[index + 1] == "_":
                # search for palatalized t_s' or t_s\
                if length - index > 3 and xsampa[index + 3] in "'\\":
                    result.append(xsampa[index : index + 4])
                    index += 4
                # otherwise, extract t_s
                else:
                    result.append(xsampa[index : index + 3])
                    index += 3
            # search for I\, s`, and z`, and all other palatal consonants
            elif xsampa[index + 1] in "`\\'":
                result.append(xsampa[index : index + 2])
                index += 2

            # otherwise, just append single consonant/vowel
            else:
                result.append(xsampa[index])
                index += 1
        else:
            result.append(xsampa[index])
            index += 1

    return result


def xsampa_to_phones(xsampa: str):
    tokens = tokenize_xsampa(xsampa)
    tokens = ["#", *tokens, "#"]

    window_size = 3
    for i in range(len(tokens) - window_size + 1):
        yield WickelPhone(*tokens[i : i + window_size])


# it is unclear whether k', g', x', and t_s' are phonemes in their own right
# parameter include toggles whether to transcribe ки as /k'i/ vs. /ki/
def cyrillic_to_xsampa(cyrillic: str, include=False):
    # predefined values
    # dict for converting simple cyrillic letters to x-sampa
    cyrillic_list = list("абвгдеёжзийклмнопрстуфхцчшщъьыэюя")
    xsampa = (
        list("abvgdeo")
        + ["z`"]
        + list("zijklmnoprstufx")
        + ["t_s", "t_s\\", "s`", "s`t_s\\", "", ""]
        + list("ieua")
    )
    alphabet = defaultdict(lambda: None, zip(cyrillic_list, xsampa))

    # strings holding different hard/soft consonants/vowels
    can_palatalize = "пбтдмнфвсзлр"
    hard_sounds = "кгшжцхчщ"
    soft_vowels = "яеиёюь"

    # initialization before loop
    result = ""
    lowered = cyrillic.lower()
    index = 0
    length = len(cyrillic)
    previous_char = " "

    while index < length:
        current_char = lowered[index]
        # check for space
        if current_char == " ":
            result += " "
        # check if current char is a palatal vowel or ь
        elif current_char in soft_vowels:
            # check if previous char is in can_palatalize
            if previous_char in can_palatalize:
                # append a ' followed by vowel in alphabet dict
                result += "'" + alphabet[current_char]
            elif previous_char in hard_sounds:
                # append just the vowel in alphabet dict when following a hard sound
                result += alphabet[current_char]
            elif current_char == "и":
                # append just "i" in case of и
                result += "i"
            else:
                # otherwise, append j followed by vowel in alphabet dict
                result += "j" + alphabet[current_char]
        else:
            # otherwise, just append letter in alphabet dict
            result += alphabet[current_char] or ""

        previous_char = current_char
        index += 1

    return result


feature_dict = {
    PhonemeType.Stop: set(stop),
    PhonemeType.Continuous: set(continuous),
    PhonemeType.Vowel: set(vowel),
    PhonemeManner.OralFricativeHigh: set(oral).union(set(fricative), set(high)),
    PhonemeManner.NasalLiquidSemiLow: set(nasal).union(set(liquid_semi), set(low)),
    PhonemePosition.Front: set(front),
    PhonemePosition.Middle: set(middle),
    PhonemePosition.Back: set(back),
    PhonemeVoice.Unvoiced: set(unvoiced),
    PhonemeVoice.Voiced: set(voiced),
    PhonemePalatalization.Hard: set(hard),
    PhonemePalatalization.Soft: set(soft),
}


def features_to_xsampa(features: list[SegmentFeatures]):
    result = ""
    for feature in features:
        phon = feature_dict[feature.phontype]
        manner = feature_dict[feature.manner]
        pos = feature_dict[feature.position]
        voice = feature_dict[feature.voice]
        palatal = feature_dict[feature.palatal]

        xsampa = phon.intersection(manner, pos, voice, palatal)
        if xsampa:
            result += list(xsampa)[0]

    return result


if __name__ == "__main__":
    np.set_printoptions(linewidth=80, formatter={
        'object': (lambda e: e.name)
    })
    xsampa = cyrillic_to_xsampa("язы́к")
    for wp in xsampa_to_phones(xsampa):
        print(f"-- activated features for Wickelphone {wp} --")
        print(wp.activating_features())