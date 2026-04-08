"""Shared phoneme articulatory constants.

Single source of truth for phoneme -> (manner, place, voice) mappings.
Both data manifest generation and model symbolic logic should import from here.
"""

from typing import Dict, Tuple


PHONEME_ARTICULATORY: Dict[str, Tuple[str, str, str]] = {
    # Stops
    "P": ("stop", "bilabial", "voiceless"),
    "B": ("stop", "bilabial", "voiced"),
    "T": ("stop", "alveolar", "voiceless"),
    "D": ("stop", "alveolar", "voiced"),
    "K": ("stop", "velar", "voiceless"),
    "G": ("stop", "velar", "voiced"),
    # Fricatives
    "F": ("fricative", "labiodental", "voiceless"),
    "V": ("fricative", "labiodental", "voiced"),
    "TH": ("fricative", "dental", "voiceless"),
    "DH": ("fricative", "dental", "voiced"),
    "S": ("fricative", "alveolar", "voiceless"),
    "Z": ("fricative", "alveolar", "voiced"),
    "SH": ("fricative", "postalveolar", "voiceless"),
    "ZH": ("fricative", "postalveolar", "voiced"),
    "HH": ("fricative", "glottal", "voiceless"),
    # Affricates
    "CH": ("affricate", "postalveolar", "voiceless"),
    "JH": ("affricate", "postalveolar", "voiced"),
    # Nasals
    "M": ("nasal", "bilabial", "voiced"),
    "N": ("nasal", "alveolar", "voiced"),
    "NG": ("nasal", "velar", "voiced"),
    # Liquids
    "L": ("liquid", "alveolar", "voiced"),
    "R": ("liquid", "alveolar", "voiced"),
    # Glides
    "W": ("glide", "labio-velar", "voiced"),
    "Y": ("glide", "palatal", "voiced"),
    # Vowels and diphthongs
    "IY": ("vowel", "front", "voiced"),
    "IH": ("vowel", "front", "voiced"),
    "EH": ("vowel", "front", "voiced"),
    "EY": ("vowel", "front", "voiced"),
    "AE": ("vowel", "front", "voiced"),
    "AA": ("vowel", "back", "voiced"),
    "AO": ("vowel", "back", "voiced"),
    "OW": ("vowel", "back", "voiced"),
    "UH": ("vowel", "back", "voiced"),
    "UW": ("vowel", "back", "voiced"),
    "AH": ("vowel", "central", "voiced"),
    "ER": ("vowel", "central", "voiced"),
    "AX": ("vowel", "central", "voiced"),
    "IX": ("vowel", "central", "voiced"),
    "AXR": ("vowel", "central", "voiced"),
    "AY": ("diphthong", "front", "voiced"),
    "AW": ("diphthong", "back", "voiced"),
    "OY": ("diphthong", "back", "voiced"),
}

# Manifest uses tuple format.
PHONEME_DETAILS: Dict[str, Tuple[str, str, str]] = PHONEME_ARTICULATORY

# Model/explainability use dict feature format.
PHONEME_FEATURES: Dict[str, Dict[str, str]] = {
    ph: {"manner": m, "place": p, "voice": v}
    for ph, (m, p, v) in PHONEME_ARTICULATORY.items()
}
