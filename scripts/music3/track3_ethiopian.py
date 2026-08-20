import sys
sys.path.insert(0, "/Users/sj/code/h3_scratch")
from music3 import run

caption = (
    "Global Metadata: Ethio-groove, Ethio-jazz influenced traditional Ethiopian music. "
    "112 BPM, Ethiopian pentatonic Tizita-inspired minor mode, hypnotic and soulful, "
    "warm nostalgic groove with a slow-burning, smoky late-night club energy.\n\n"
    "Vocal Details: Warm expressive male Amharic-style vocal, melismatic ornamented "
    "phrasing typical of azmari singing, soulful and yearning delivery, call-and-response "
    "with a small backing chorus on the hook.\n\n"
    "Arrangement: Krar (six-string lyre) picking out a hypnotic pentatonic riff, washint "
    "flute answering the vocal lines, deep kebero hand-drum groove with a loose swing, "
    "warm analog Rhodes-style keys, a saxophone playing a smoky Ethio-jazz countermelody. "
    "Intro: krar riff alone, kebero drum enters. Verses: full groove under the vocal. "
    "Chorus: saxophone and backing chorus join, the riff opens up. Outro: sax solo "
    "trailing off as the krar riff loops and fades."
)

lyrics = (
    "[intro]\n"
    "(krar riff, drum enters)\n"
    "[verse]\n"
    "(warm Amharic-style vocal, melismatic and yearning)\n"
    "[chorus]\n"
    "(vocal and backing chorus, call and response)\n"
    "[instrumental]\n"
    "(saxophone countermelody over the groove)\n"
    "[outro]\n"
    "(sax solo fading, krar riff loops out)"
)

run("ethiopian_groove", caption, lyrics, seconds=30.0, seed=303)
