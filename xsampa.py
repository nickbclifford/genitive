from enum import Enum
from itertools import chain

# phoneme types
vowel = ["a", "e", "i", "I\\", "o", "u"]
stop = list(chain.from_iterable((sound, sound + "'") for sound in "pbtdkgmn"))
continuous = list(chain.from_iterable((sound, sound + "'") for sound in "f|v|l|s|z|r|t_s|x".split("|")))
continuous.extend(["s`", "t_s\\", "z`", "j"])

# phoneme manners
high = ["I\\", "i", "u"]
low = ["e", "a", "o"]
nasal = list(chain.from_iterable((sound, sound + "'") for sound in "mn"))
liquid_semi = list(chain.from_iterable((sound, sound + "'") for sound in "l|t_s|r|x".split("|")))
liquid_semi.append("j")
fricative = list(chain.from_iterable((sound, sound + "'") for sound in "fvsz"))
fricative.extend(["s`", "t_s\\", "z`"])
oral = list(chain.from_iterable((sound, sound + "'") for sound in "pbtdkg"))

# phoneme positions
front = list(chain.from_iterable((sound, sound + "'") for sound in "pbmfvl"))
front.extend(["I\\", "i", "e"])
middle = list(chain.from_iterable((sound, sound + "'") for sound in "t|d|n|s|z|t_s|r".split("|")))
middle.append("a")
back = list(chain.from_iterable((sound, sound + "'") for sound in "kgx"))
back.extend(["s`", "t_s\\", "z`", "j", "u", "o"])

# phoneme voicedness
unvoiced = list(chain.from_iterable((sound, sound + "'") for sound in "p|f|t|s|t_s|k|x".split("|")))
unvoiced.extend(["s`", "t_s\\"])
voiced = list(chain.from_iterable((sound, sound + "'") for sound in "bmvldnzrg"))
voiced.extend(["z`", "j", "a", "I\\", "i", "e", "o", "u"])

# phoneme palatalization
hard = [sound for sound in "p|f|b|m|v|l|I\\|t|s|t_s|d|n|z|r|a|k|s`|x|g|z`|u|o".split("|")]
soft = [sound + "'" for sound in "pfbmvltsdnzrkgx"]
soft.extend(["i", "e", "t_s\\", "j", "t_s'"])


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
    Oral = 0
    Nasal = 1
    Fricative = 2
    LiquidSemivowel = 3
    High = 4
    Low = 5

    def from_xsampa(phoneme):
        if phoneme in high:
            return PhonemeManner.High
        
        if phoneme in low:
            return PhonemeManner.Low

        if phoneme in nasal:
            return PhonemeManner.Nasal

        if phoneme in liquid_semi:
            return PhonemeManner.LiquidSemivowel

        if phoneme in fricative:
            return PhonemeManner.Fricative

        if phoneme in oral:
            return PhonemeManner.Oral

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

class WickelFeature:
    def __init__(self, phoneme, boundary=False):
        if not boundary:
            self.phontype = PhonemeType.from_xsampa(phoneme)
            self.manner = PhonemeManner.from_xsampa(phoneme)
            self.position = PhonemePosition.from_xsampa(phoneme)
            self.voice = PhonemeVoice.from_xsampa(phoneme)
            self.palatal = PhonemePalatalization.from_xsampa(phoneme)
            self.boundary = False
        else:
            self.phontype = None
            self.manner = None
            self.position = None
            self.voice = None
            self.palatal = None
            self.boundary = True

    def __str__(self):
        return f'Type: {self.phontype}\nManner: {self.manner}\nPosition: {self.position}\nVoice: {self.voice}\nPalatalization: {self.palatal}\nBoundary: {self.boundary}'

class WickelPhone:
    def __init__(self, first, second, third):
        self.phones = [first, second, third]


def tokenize_xsampa(xsampa):
    result = []
    while xsampa:
        # search for joint consonants t_s, t_s', t_s\
        if len(xsampa) > 1:
            if xsampa[1] == '_':
                # search for palatalized t_s' or t_s\
                if len(xsampa) > 3 and xsampa[3] in "'\\":
                    result.append(xsampa[:4])
                    xsampa = xsampa[4:]
                # otherwise, extract t_s
                else:
                    result.append(xsampa[:3])
                    xsampa = xsampa[3:]
            # search for I\, s`, and z`, and all other palatal consonants
            elif xsampa[1] in "`\\'":
                result.append(xsampa[:2])
                xsampa = xsampa[2:]

            # otherwise, just append single consonant/vowel
            else:
                result.append(xsampa[0])
                xsampa = xsampa[1:]
        else:
            result.append(xsampa)
            xsampa = ''

    return result

def xsampa_to_features(xsampa):
    tokens = tokenize_xsampa(xsampa)
    return list(map(WickelFeature, tokens))
