import json
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from xsampa import cyrillic_to_xsampa
from wickel import (
    WickelPhone,
    xsampa_to_phones,
    wickelfeatures,
    feature_index_map,
    build_word_data,
)


def activations_for_phone(phone: WickelPhone):
    activations = np.zeros(len(wickelfeatures), dtype=np.bool_)

    for feature in phone.activating_features():
        # TODO: this is probably vectorizable somehow
        for blurred in feature.blur():
            # we might have blurred into a feature we're not checking for
            if index := feature_index_map.get(blurred):
                activations[index] = True

    return activations


def activations_for_word(phones: list[WickelPhone]):
    activations = np.zeros(len(wickelfeatures), dtype=np.bool_)

    for phone in phones:
        activations = np.bitwise_or(activations, activations_for_phone(phone))

    return activations.astype(np.float32)


class Net(nn.Module):
    def __init__(self, output_multiplier=1):
        super(Net, self).__init__()

        self.layer = nn.Linear(
            in_features=len(wickelfeatures),
            out_features=output_multiplier * len(wickelfeatures),
            bias=True,
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

net_gen_sg = Net().to(device)
net_gen_pl = Net().to(device)
net_both = Net(output_multiplier=2).to(device)

loss_criterion = nn.L1Loss().to(device)

BATCH_SIZE = 1024

def run_training(model, mapping_pairs, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=15.0)

    for epoch in range(num_epochs):
        for input_rep, target_rep in DataLoader(mapping_pairs, batch_size=BATCH_SIZE):
            input_rep = input_rep.to(device)
            target_rep = target_rep.to(device)

            optimizer.zero_grad()
            output = model(input_rep)
            loss = loss_criterion(output, target_rep)
            loss.backward()
            optimizer.step()

        print(f"loss at epoch {epoch}: {loss.item():.6f}")


def build_activations():
    if os.path.exists("activations.json"):
        print("using cached activations")
        with open("activations.json") as f:
            words = json.load(f)
    else:
        print("building word activations")
        words = {}
        for word, nom, sg, pl in build_word_data():
            nom_act = activations_for_word(nom)
            sg_act = activations_for_word(sg)
            pl_act = activations_for_word(pl)

            print(f"built activations for {word}")

            words[word] = {
                "nom_sg": {
                    "phones": [str(p) for p in nom],
                    "activations": nom_act.tolist(),
                },
                "gen_sg": {
                    "phones": [str(p) for p in sg],
                    "activations": sg_act.tolist(),
                },
                "gen_pl": {
                    "phones": [str(p) for p in pl],
                    "activations": pl_act.tolist(),
                },
            }
        with open("activations.json", "w") as f:
            json.dump(words, f)

    for acts in words.values():
        yield torch.FloatTensor(acts["nom_sg"]["activations"]), torch.FloatTensor(
            acts["gen_sg"]["activations"]
        ), torch.FloatTensor(acts["gen_pl"]["activations"])


if __name__ == "__main__":
    sg_maps = []
    pl_maps = []
    both_maps = []

    for nom_act, sg_act, pl_act in build_activations():
        sg_maps.append((nom_act, sg_act))
        pl_maps.append((nom_act, pl_act))
        both_maps.append((nom_act, torch.cat((sg_act, pl_act))))

    # TODO: start train/test splitting to check for overfitting

    print("beginning training...")
    run_training(net_gen_sg, sg_maps, 100)
    torch.save(net_gen_sg.state_dict(), "sg.pt")
    print("trained singular model")
    run_training(net_gen_pl, pl_maps, 100)
    torch.save(net_gen_pl.state_dict(), "pl.pt")
    print("trained plural model")
    run_training(net_both, both_maps, 100)
    torch.save(net_gen_sg.state_dict(), "both.pt")

    test_word = "жена"
    test_input = torch.from_numpy(
        activations_for_word(xsampa_to_phones(cyrillic_to_xsampa(test_word)))
    ).to(device)

    predicted_sg = net_gen_sg(test_input)
    print("Singular model output:", predicted_sg)
    predicted_pl = net_gen_pl(test_input)
    print("Plural model output:", predicted_pl)
    predicted_both = net_both(test_input)
    print("Dual model output:", predicted_both)

    both_list = predicted_both.tolist()
    with open("predictions.json", "w") as f:
        json.dump(
            {
                "sg": predicted_sg.tolist(),
                "pl": predicted_pl.tolist(),
                "both_sg": both_list[: len(wickelfeatures)],
                "both_pl": both_list[len(wickelfeatures) :],
            },
            f,
        )
