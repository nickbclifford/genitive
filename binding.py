import functools

import numpy as np

from wickel import (
    WickelPhone,
    build_word_data,
    wickelfeatures,
    feature_index_map,
)

all_phones = set[WickelPhone]()
for _, nom, gen_sg, gen_pl in build_word_data():
    for phone in nom:
        all_phones.add(phone)
    for phone in gen_sg:
        all_phones.add(phone)
    for phone in gen_pl:
        all_phones.add(phone)
all_phones = sorted(list(all_phones)) # sort to ensure deterministic ordering, since sets are unordered

input_features = len(wickelfeatures)
output_features = len(all_phones)

filtered_activations = [
    np.array([feature_index_map[a] for a in p.filtered_activations()])
    for p in all_phones
]

feature_to_phones = list(set() for _ in range(input_features))
for phone, features in enumerate(filtered_activations):
    for feature in features:
        feature_to_phones[feature].add(phone)
feature_to_phones = np.array(feature_to_phones, dtype=np.object_)

WORD_SIZE = 10


def predict_word(prediction):
    i = np.array(prediction)

    import time

    output_phones: list[WickelPhone] = []

    output = np.zeros([WORD_SIZE, output_features])
    for t in range(WORD_SIZE):
        start = time.time()

        prev_time = np.ones(output_features) if t < 1 else output[t - 1]

        for k in range(output_features):
            this_phone = all_phones[k]
            this_left, this_center, _ = this_phone.phones
            if not output_phones:
                if this_left != "#":
                    output[t, k] = 0
                    continue
            else:
                _, prev_center, prev_right = output_phones[-1].phones
                if this_left != prev_center or this_center != prev_right:
                    output[t, k] = 0
                    continue

            js = filtered_activations[k]
            ls = feature_to_phones[js]
            ls = functools.reduce(lambda s, t: s | t, ls)
            ls = np.array(list(ls))

            numerator = np.sum(i[js] * prev_time[k])
            denominator = np.sum(prev_time[ls])

            output[t, k] = numerator / denominator

            print(f"{k + 1}/{output_features}", end="\r")
        
        end = time.time()

        next_phone = all_phones[np.argmax(output[t])]

        output_phones.append(next_phone)
        # print(f"predicted phone {t + 1} in {end - start:.2f}s - {next_phone}")

        if next_phone.phones[2] == '#':
            break

    return output_phones


if __name__ == "__main__":
    import json

    with open("predictions.json") as f:
        p = json.load(f)

    with open("decoded.json", "w") as f:
        out = {}
        for k, v in p.items():
            out[k] = [str(p) for p in predict_word(v)]
            print(f"{k}: {out[k]}")
        json.dump(out, f)
