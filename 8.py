import collections
import collections.abc
# Monkey-patch collections to support older packages referencing collections.Iterable
collections.Iterable = collections.abc.Iterable

from inltk.inltk import setup, get_similar_sentences

setup('kn')

def paraphrase_text_kn(
    text,
    num_variants=5,
    degree=0.2
):
    return get_similar_sentences(
        text,
        num_variants,
        'kn',
        degree_of_aug=degree
    )

if __name__ == "__main__":
    input_sentence = "ನಾನು ಇಂದು ಬಹಳ ಸಂತೋಷವಾಗಿದ್ದೇನೆ"

    print(
        f"ಮೂಲ ವಾಕ್ಯ : {input_sentence}\n"
    )

    paraphrases = paraphrase_text_kn(
        input_sentence,
        num_variants=6,
        degree=0.2
    )

    print("ಪರ್ಯಾಯ ವಾಕ್ಯಗಳು:")

    for i, sent in enumerate(
        paraphrases,
        1
    ):
        print(f"{i}. {sent}")
