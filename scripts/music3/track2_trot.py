import sys
sys.path.insert(0, "/Users/sj/code/h3_scratch")
from music3 import run

caption = (
    "Global Metadata: Korean trot (트로트), classic ppongjjak style. 132 BPM, "
    "G major, bright and bouncy duple-feel oom-pah rhythm. Cheerful, festive, nostalgic "
    "warmth, the feeling of a lively countryside festival stage.\n\n"
    "Vocal Details: Energetic female trot vocal with heavy, fast vibrato and classic "
    "kkeokki (꾬기) melodic ornamentation on held notes, bright forward belted tone, "
    "confident and playful delivery, occasional spoken ad-libs between phrases.\n\n"
    "Arrangement: Bouncy synth accordion and organ oom-pah bassline, bright muted trumpet "
    "stabs answering the vocal, tambourine and a tight electronic trot drum-machine beat "
    "with a signature swung snare. Intro: short trumpet fanfare into the drum-machine "
    "beat. Verses: accordion oom-pah carries the groove under the vocal. Chorus: full "
    "band, trumpet countermelody, bigger and brighter. Outro: one last trumpet flourish "
    "and a stopped ending."
)

lyrics = (
    "[intro]\n"
    "(trumpet fanfare)\n"
    "[verse]\n"
    "오늘도 하루가 시작되었네\n"
    "황금같은 행복이 찾아오네\n"
    "[chorus]\n"
    "아자자 좀종좀종 기뻐게 놀자구나\n"
    "인생은 한 번이야 흥여우자\n"
    "[verse]\n"
    "길거리에 웃음이 가득하네\n"
    "[chorus]\n"
    "아자자 좀종좀종 기뻐게 놀자구나\n"
    "인생은 한 번이야 흥여우자\n"
    "[outro]\n"
    "(trumpet flourish, stopped ending)"
)

run("korean_trot", caption, lyrics, seconds=30.0, seed=202)
