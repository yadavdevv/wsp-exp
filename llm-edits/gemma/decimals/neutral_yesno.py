"""40 general yes/no questions used ONLY to fit the edit (target: no change). Disjoint from facts_yesno.py (the test)."""
NEUTRAL = ["Is Rome in Italy?", "Is milk white?", "Is the sky green?", "Do dogs bark?", "Is salt sweet?",
           "Is Berlin the capital of Germany?", "Is a tomato blue?", "Do people sleep at night?", "Is the Nile in Africa?",
           "Is a mouse bigger than an elephant?", "Is Kolkata a city in India?", "Do fish have wings?", "Is fire hot?",
           "Is Saturn a planet?", "Is a rose a flower?", "Is Canada in Africa?", "Do apples grow on trees?",
           "Is the Sahara a forest?", "Is Hindi spoken in India?", "Is ten an odd number?", "Is snow cold?",
           "Is the Amazon a river?", "Do cars run on water?", "Is a lion a cat?", "Is Spain in Asia?",
           "Is iron magnetic?", "Do penguins live at the North Pole?", "Is coffee a drink?", "Is a cube round?",
           "Is Chennai on the coast?", "Is Jupiter smaller than the Moon?", "Do children go to school?",
           "Is an hour sixty minutes?", "Is the Arctic warm?", "Is a piano a musical instrument?", "Is a carrot a fruit?",
           "Is Kenya in Africa?", "Do horses have hooves?", "Is glass made from sand?", "Is a kilometre shorter than a metre?"]

# v3 additions (still disjoint from the test questions in facts_yesno.py)
_CAP = [("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Russia", "Moscow"),
        ("China", "Beijing"), ("Spain", "Madrid"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Greece", "Athens"),
        ("Nepal", "Kathmandu"), ("Canada", "Ottawa"), ("Thailand", "Bangkok"), ("Norway", "Oslo"), ("Iran", "Tehran")]
_ANIMAL = [("cats", "whiskers", True), ("cows", "wings", False), ("elephants", "trunks", True), ("fish", "gills", True),
           ("snakes", "fur", False), ("ducks", "feathers", True), ("dogs", "tails", True), ("horses", "horns", False),
           ("spiders", "eight legs", True), ("frogs", "feathers", False), ("camels", "humps", True), ("whales", "legs", False)]
_CMP = ["Is an elephant bigger than a mouse?", "Is an ant bigger than a horse?", "Is the Sun hotter than ice?",
        "Is a car faster than a bicycle?", "Is a snail faster than a cheetah?", "Is a mountain taller than a house?",
        "Is a child older than their grandmother?", "Is a minute longer than an hour?", "Is lead heavier than a feather?",
        "Is a village bigger than a country?", "Is Asia larger than Australia?", "Is a pebble heavier than a boulder?",
        "Is summer warmer than winter?", "Is a kitten smaller than a tiger?", "Is a river wider than an ocean?",
        "Is the Earth closer to the Sun than Neptune?", "Is a second shorter than a day?", "Is a bus longer than a pen?"]
NEUTRAL3 = (NEUTRAL
            + [f"Is {c} the capital of {k}?" for k, c in _CAP]
            + [f"Is {_CAP[(i + 3) % len(_CAP)][1]} the capital of {k}?" for i, (k, _) in enumerate(_CAP)]
            + [f"Do {a} have {p}?" for a, p, _ in _ANIMAL]
            + _CMP)
