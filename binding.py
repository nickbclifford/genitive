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
all_phones = list(all_phones)

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

    output = np.zeros([WORD_SIZE, output_features])
    for t in range(WORD_SIZE):
        prev_time = np.ones(output_features) if t < 1 else output[t - 1]

        for k in range(output_features):
            js = filtered_activations[k]
            ls = feature_to_phones[js]
            ls = functools.reduce(lambda s, t: s | t, ls)
            ls = np.array(list(ls))

            numerator = np.sum(i[js] * prev_time[k])
            denominator = np.sum(prev_time[ls])

            output[t, k] = numerator / denominator
        print(f"calculated all of time {t}")

    return output


if __name__ == "__main__":
    import json

    with open("predictions.json") as f:
        p = json.load(f)

    with open("decoded.json", "w") as f:
        out = {}
        for k, v in p.items():
            decoded = predict_word(v)
            out[k] = [str(all_phones[i]) for i in np.argmax(decoded, axis=1)]
            print(f"{k}: {out[k]}")
            out[k + "_weights"] = decoded.tolist()
        json.dump(out, f)
