from thesis.constants import StringEnum

LABEL_MAPPING = {
    "n": 0,
    "p": 1,
}

REVERSE_LABEL_MAPPING = {
    0: "n",
    1: "p",
}

class Dicova2AudioTypes(StringEnum):
    BREATHING = "breathing"
    COUGH = "cough"
    SPEECH = "speech"

# From Paper: All audio recordings were re-sampled to 44.1 kHz
SAMPLING_RATE = 44100

VALID_FOLDS = [0, 1, 2, 3, 4]
