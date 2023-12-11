import itertools
import json
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from wickel import (
    WickelPhone,
    build_word_data,
    wickelfeatures,
)

all_phones = set[WickelPhone]()
for _, nom, gen_sg, gen_pl in build_word_data():
    for phone in nom:
        all_phones.add(phone)
    for phone in gen_sg:
        all_phones.add(phone)
    for phone in gen_pl:
        all_phones.add(phone)
all_phones = sorted(
    list(all_phones)
)  # sort to ensure deterministic ordering, since sets are unordered
phones_index_map = { str(p): i for i, p in enumerate(all_phones) }


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer = nn.Linear(
            in_features=len(wickelfeatures) + len(all_phones),
            out_features=len(all_phones),
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

dec_model = Decoder().to(device)

loss_criterion = nn.L1Loss().to(device)

BATCH_SIZE = 16 * 1024

def run_training(model, mapping_pairs, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        for i, (input_rep, target_rep) in enumerate(DataLoader(mapping_pairs, batch_size=BATCH_SIZE)):
            optimizer.zero_grad()
            output = model(input_rep.to(device))
            loss = loss_criterion(output, target_rep.to(device))
            loss.backward()
            optimizer.step()
            print(f"trained {((i + 1) * BATCH_SIZE)}/{len(mapping_pairs)}", end="\r")

        print(f"loss at epoch {epoch}: {loss.item():.6f}")


def build_training_data():
    if not os.path.exists("activations.json"):
        import model
        model.build_activations()

    with open("activations.json") as f:
        words = json.load(f)
    
    for cases in words.values():
        for case in cases.values():
            last_phone = torch.zeros([len(all_phones)])

            for phone in case["phones"]:
                phone_pattern = torch.zeros([len(all_phones)])
                phone_pattern[phones_index_map[phone]] = 1
                yield (
                    torch.cat((torch.FloatTensor(case["activations"]), last_phone)),
                    phone_pattern
                )

                last_phone = phone_pattern

if __name__ == "__main__":
    if os.path.exists("decoder.pt"):
        dec_model.load_state_dict(torch.load("decoder.pt"))
        dec_model.eval()
    else:
        print("beginning training")
        run_training(dec_model, build_training_data(), 50)
        torch.save(dec_model.state_dict(), "decoder.pt")


    with open("predictions.json") as f:
        p = json.load(f)

    phones = []
    last_phone = torch.zeros([len(all_phones)])
    acts = torch.FloatTensor(p["sg"])

    while len(phones) < 10:
        next_phone = dec_model(torch.cat((acts, last_phone)).to(device))
        idx = torch.argmax(next_phone)
        phone = all_phones[idx]

        if phone.phones[2] == '#':
            break

        phones.append(phone)

        last_phone = torch.zeros([len(all_phones)])
        last_phone[idx] = 1

    print(phones)
