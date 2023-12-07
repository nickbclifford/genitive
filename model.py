import csv
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from xsampa import (
    PhonemeManner,
    PhonemePalatalization,
    PhonemePosition,
    PhonemeType,
    PhonemeVoice,
    WordBoundary,
    cyrillic_to_xsampa,
    tokenize_xsampa,
    xsampa_to_phones,
)


all_features = [
    *PhonemeType,
    *PhonemeManner,
    *PhonemePosition,
    *PhonemeVoice,
    *PhonemePalatalization,
]
with_boundary = all_features + [WordBoundary.Boundary]


def all_wickelfeatures():
    for pre, central, post in itertools.product(
        with_boundary, all_features, with_boundary
    ):
        feature = np.array([pre, central, post])

        if (
            pre == WordBoundary.Boundary
            or post == WordBoundary.Boundary
            or type(pre) is type(post)
        ):
            yield feature


wickelfeatures = list(all_wickelfeatures())


# TODO: it'd be nice if we could reuse this class for the one-model-two-cases output
class Net(nn.Module):
    def __init__(self, output_multiplier=1):
        super(Net, self).__init__()

        self.layer = nn.Linear(
            in_features=len(wickelfeatures),
            out_features=output_multiplier * len(wickelfeatures),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x


net_gen_sg = Net()
net_gen_pl = Net()

loss_criterion = nn.L1Loss()


def run_training(model, mapping_pairs, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=10.0)

    for epoch in range(num_epochs):
        for input_rep, target_rep in mapping_pairs:
            # TODO: this is copied from the problem set, do we actually need it?
            input_rep = input_rep.view(1, -1)
            target_rep = target_rep.view(1, -1)

            optimizer.zero_grad()
            output = model(input_rep)
            loss = loss_criterion(output, target_rep)
            loss.backward()
            optimizer.step()


def build_word_data():
    with open("declensions.csv") as file:
        for row in csv.DictReader(file):
            yield (
                xsampa_to_phones(cyrillic_to_xsampa(row["title"])),
                xsampa_to_phones(cyrillic_to_xsampa(row["gen_sg"])),
                xsampa_to_phones(cyrillic_to_xsampa(row["gen_pl"])),
            )


# TODO: build activation patterns for individual words
# turn into mapping pairs of activations over *all* wickelfeatures


BLUR_PROBABILILTY = 0.9


def blur_wickelfeature(feature):
    yield feature

    for pre in random.sample(with_boundary, random.randint(0, len(with_boundary))):
        if random.random() > BLUR_PROBABILILTY:
            yield np.array([pre, feature[1], feature[2]])

    for post in random.sample(with_boundary, random.randint(0, len(with_boundary))):
        if random.random() > BLUR_PROBABILILTY:
            yield np.array([feature[0], feature[1], post])


if __name__ == "__main__":
    np.set_printoptions(linewidth=80, formatter={"object": (lambda e: e.name)})
    for feature in all_wickelfeatures():
        print(feature)
