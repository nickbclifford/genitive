import csv
import itertools
import random
import os.path

import numpy as np

from xsampa import (
    PhonemeManner,
    PhonemePalatalization,
    PhonemePosition,
    PhonemeType,
    PhonemeVoice,
    WordBoundary,
    cyrillic_to_xsampa,
    tokenize_xsampa,
    feature_dict,
)


def encode_feature(feature):
    # 3 bits necessary to encode feature type
    # 2 bits necessary to encode feature value

    if type(feature) is PhonemeType:
        f_type = 0
    elif type(feature) is PhonemeManner:
        f_type = 1
    elif type(feature) is PhonemePosition:
        f_type = 2
    elif type(feature) is PhonemeVoice:
        f_type = 3
    elif type(feature) is PhonemePalatalization:
        f_type = 4
    else:  # word boundary
        f_type = 5

    value = feature.value

    return (f_type << 2) | value


def decode_feature(encoded):
    value = encoded & 0b11
    f_type = encoded >> 2

    match f_type:
        case 0:
            enum = PhonemeType
        case 1:
            enum = PhonemeManner
        case 2:
            enum = PhonemePosition
        case 3:
            enum = PhonemeVoice
        case 4:
            enum = PhonemePalatalization
        case _:
            enum = WordBoundary

    return enum(value)


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


all_features = [
    *PhonemeType,
    *PhonemeManner,
    *PhonemePosition,
    *PhonemeVoice,
    *PhonemePalatalization,
]
with_boundary = all_features + [WordBoundary.Boundary]


BLUR_PROBABILILTY = 0.9


class WickelFeature:
    def __init__(self, first, second, third):
        self.features = [first, second, third]

    def blur(self):
        yield self

        for pre in random.sample(with_boundary, random.randint(0, len(with_boundary))):
            if random.random() > BLUR_PROBABILILTY:
                yield WickelFeature(pre, self.features[1], self.features[2])

        for post in random.sample(with_boundary, random.randint(0, len(with_boundary))):
            if random.random() > BLUR_PROBABILILTY:
                yield WickelFeature(self.features[0], self.features[1], post)

    def __repr__(self):
        return "<" + ", ".join(f.name for f in self.features) + ">"

    def __hash__(self):
        return self.encode()
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    def encode(self):
        pre, central, post = [encode_feature(f) for f in self.features]
        return (pre << 16) | (central << 8) | post

    @classmethod
    def decode(cls, encoded):
        pre = encoded >> 16
        central = (encoded >> 8) & 0xFF
        post = encoded & 0xFF

        return cls(decode_feature(pre), decode_feature(central), decode_feature(post))
    

def all_wickelfeatures():
    for pre, central, post in itertools.product(
        with_boundary, all_features, with_boundary
    ):
        feature = WickelFeature(pre, central, post)

        if (
            pre == WordBoundary.Boundary
            or post == WordBoundary.Boundary
            or type(pre) is type(post)
        ):
            yield feature


wickelfeatures = list(all_wickelfeatures())
feature_index_map = {f: i for i, f in enumerate(wickelfeatures)}


class WickelPhone:
    def __init__(self, first: str, second: str, third: str):
        self.phones = [first, second, third]

    def __hash__(self):
        return hash(self.__repr__())
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
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

        for vec in np.transpose(cols):
            yield WickelFeature(*vec)

    def filtered_activations(self):
        for feature in self.activating_features():
            pre, _, post = feature.features
            if (
                pre == WordBoundary.Boundary
                or post == WordBoundary.Boundary
                or type(pre) is type(post)
            ):
                yield feature


def build_word_data():
    if not os.path.exists("declensions.csv"):
        import scraper
        import asyncio

        asyncio.run(scraper.main())

    with open("declensions.csv") as file:
        for row in csv.DictReader(file):
            yield (
                row["title"],
                xsampa_to_phones(cyrillic_to_xsampa(row["title"])),
                xsampa_to_phones(cyrillic_to_xsampa(row["gen_sg"])),
                xsampa_to_phones(cyrillic_to_xsampa(row["gen_pl"])),
            )


def xsampa_to_phones(xsampa: str):
    tokens = tokenize_xsampa(xsampa)
    tokens = ["#", *tokens, "#"]

    window_size = 3
    for i in range(len(tokens) - window_size + 1):
        yield WickelPhone(*tokens[i : i + window_size])


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
