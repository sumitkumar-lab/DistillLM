import json, random

def augment(text):
    noise = [
        text,
        text.lower(),
        text.replace("the ", ""),
        text.replace("on", "up"),
    ]
    return list(set(noise))

def build():
    with open("data/raw/intents_seed.json") as f:
        data = json.load(f)

    samples = []
    for intent, phrases in data.items():
        for p in phrases:
            for aug in augment(p):
                samples.append({
                    "text": aug,
                    "label": intent
                })

    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train, test = samples[:split], samples[split:]

    json.dump(train, open("data/processed/train.json", "w"), indent=2)
    json.dump(test, open("data/processed/test.json", "w"), indent=2)

if __name__ == "__main__":
    build()
