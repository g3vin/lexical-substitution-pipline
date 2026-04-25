#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-example_outputs.json}"
TOP_K="${TOP_K:-10}"
T5_MODEL="${T5_MODEL:-google/flan-t5-base}"
SUBSTITUTE_MODEL="${SUBSTITUTE_MODEL:-./substitute-roberta}"
RANKER_CONFIG="${RANKER_CONFIG:-./ranker_config.json}"
LEARNED_RANKER_MODEL="${LEARNED_RANKER_MODEL:-./learned_ranker.pkl}"
SWORDS_PATH="${SWORDS_PATH:-./data/swords/swords-v1.1_dev.json.gz}"
GENERATION_MODE="${GENERATION_MODE:-balanced}"
ENABLE_RERANKER="${ENABLE_RERANKER:-1}"
NLI_MODEL="${NLI_MODEL:-}"
NLI_CONTRADICTION_THRESHOLD="${NLI_CONTRADICTION_THRESHOLD:-0.80}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

echo "Writing to: $OUTPUT_PATH"
echo "Mode: $GENERATION_MODE | reranker: $ENABLE_RERANKER | max examples: $MAX_EXAMPLES | top_k: $TOP_K"
if [[ -f "$LEARNED_RANKER_MODEL" ]]; then
  echo "Learned ranker: $LEARNED_RANKER_MODEL"
fi
if [[ -n "$NLI_MODEL" ]]; then
  echo "NLI contradiction filter: $NLI_MODEL @ $NLI_CONTRADICTION_THRESHOLD"
fi

python - "$OUTPUT_PATH" "$TOP_K" "$T5_MODEL" "$SUBSTITUTE_MODEL" "$RANKER_CONFIG" "$LEARNED_RANKER_MODEL" "$SWORDS_PATH" "$GENERATION_MODE" "$ENABLE_RERANKER" "$NLI_MODEL" "$NLI_CONTRADICTION_THRESHOLD" "$MAX_EXAMPLES" <<'PY'
import json
import sys
from pathlib import Path

from pipeline import LexicalSubstitutionPipeline


output_path = Path(sys.argv[1])
top_k = int(sys.argv[2])
t5_model = sys.argv[3]
substitute_model = Path(sys.argv[4])
ranker_config = Path(sys.argv[5])
learned_ranker_model = Path(sys.argv[6])
swords_path = Path(sys.argv[7])
generation_mode = sys.argv[8]
enable_reranker = sys.argv[9].lower() in {"1", "true", "yes", "on"}
nli_model = sys.argv[10] or None
nli_contradiction_threshold = float(sys.argv[11])
max_examples = int(sys.argv[12])

# a bunch of classics as training examples because why not hahaha
examples = [
    ("It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife.", "acknowledged", "Pride and Prejudice"),
    ("My good opinion once lost is lost forever.", "lost", "Pride and Prejudice"),
    ("I dearly love a laugh.", "dearly", "Pride and Prejudice"),
    ("Silly things do cease to be silly if they are done by sensible people.", "sensible", "Emma"),
    ("Vanity working on a weak head produces every sort of mischief.", "weak", "Emma"),
    ("I am no bird; and no net ensnares me.", "ensnares", "Jane Eyre"),
    ("Reader, I married him.", "married", "Jane Eyre"),
    ("I would always rather be happy than dignified.", "happy", "Jane Eyre"),
    ("Whatever our souls are made of, his and mine are the same.", "same", "Wuthering Heights"),
    ("I cannot live without my life.", "live", "Wuthering Heights"),
    ("Beware; for I am fearless, and therefore powerful.", "fearless", "Frankenstein"),
    ("Nothing is so painful to the human mind as a great and sudden change.", "sudden", "Frankenstein"),
    ("Listen to them, the children of the night.", "Listen", "Dracula"),
    ("There are darknesses in life and there are lights.", "lights", "Dracula"),
    ("We learn from failure, not from success.", "failure", "Dracula"),
    ("The books that the world calls immoral are books that show the world its own shame.", "immoral", "The Picture of Dorian Gray"),
    ("Nowadays people know the price of everything and the value of nothing.", "value", "The Picture of Dorian Gray"),
    ("The only way to get rid of a temptation is to yield to it.", "yield", "The Picture of Dorian Gray"),
    ("Call me Ishmael.", "Call", "Moby-Dick"),
    ("Whenever I find myself growing grim about the mouth, I account it high time to get to sea.", "grim", "Moby-Dick"),
    ("It is better to sleep with a sober cannibal than a drunken Christian.", "sober", "Moby-Dick"),
    ("It was the best of times, it was the worst of times.", "best", "A Tale of Two Cities"),
    ("I loved her against reason, against promise, against peace.", "loved", "Great Expectations"),
    ("Please, sir, I want some more.", "want", "Oliver Twist"),
    ("I will honour Christmas in my heart, and try to keep it all the year.", "honour", "A Christmas Carol"),
    ("God bless us, every one!", "bless", "A Christmas Carol"),
    ("Curiouser and curiouser!", "Curiouser", "Alice's Adventures in Wonderland"),
    ("We're all mad here.", "mad", "Alice's Adventures in Wonderland"),
    ("Begin at the beginning, and go on till you come to the end.", "Begin", "Alice's Adventures in Wonderland"),
    ("The time has come, the Walrus said, to talk of many things.", "talk", "Through the Looking-Glass"),
    ("There is no place like home.", "place", "The Wonderful Wizard of Oz"),
    ("The road to the City of Emeralds is paved with yellow brick.", "paved", "The Wonderful Wizard of Oz"),
    ("If you look the right way, you can see that the whole world is a garden.", "see", "The Secret Garden"),
    ("Where you tend a rose, a thistle cannot grow.", "tend", "The Secret Garden"),
    ("There is nothing half so much worth doing as simply messing about in boats.", "messing", "The Wind in the Willows"),
    ("Believe me, my young friend, there is nothing absolute except relativity.", "absolute", "The Wind in the Willows"),
    ("Fifteen men on the dead man's chest.", "dead", "Treasure Island"),
    ("I am captain here by election.", "election", "Treasure Island"),
    ("To die will be an awfully big adventure.", "die", "Peter Pan"),
    ("All children, except one, grow up.", "grow", "Peter Pan"),
    ("Tomorrow is always fresh, with no mistakes in it.", "fresh", "Anne of Green Gables"),
    ("Isn't it nice to think that tomorrow is a new day with no mistakes in it yet?", "nice", "Anne of Green Gables"),
    ("I am not afraid of storms, for I am learning how to sail my ship.", "afraid", "Little Women"),
    ("I like good strong words that mean something.", "strong", "Little Women"),
    ("You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer.", "read", "Adventures of Huckleberry Finn"),
    ("All right, then, I'll go to hell.", "go", "Adventures of Huckleberry Finn"),
    ("The elastic heart of youth cannot be compressed into one constrained shape.", "constrained", "The Adventures of Tom Sawyer"),
    ("All human wisdom is contained in these two words, Wait and Hope.", "contained", "The Count of Monte Cristo"),
    ("All for one, one for all.", "all", "The Three Musketeers"),
    ("He never went out without a book under his arm.", "went", "Les Miserables"),
    ("Finally, from so little sleeping and so much reading, his brain dried up.", "dried", "Don Quixote"),
    ("The truth may be stretched thin, but it never breaks.", "stretched", "Don Quixote"),
    ("We can know only that we know nothing.", "know", "War and Peace"),
    ("Happy families are all alike; every unhappy family is unhappy in its own way.", "unhappy", "Anna Karenina"),
    ("Pain and suffering are always inevitable for a large intelligence.", "inevitable", "Crime and Punishment"),
    ("The mystery of human existence lies not in just staying alive.", "staying", "The Brothers Karamazov"),
    ("I am a sick man; I am a spiteful man.", "spiteful", "Notes from Underground"),
    ("The game is afoot.", "afoot", "The Adventure of the Abbey Grange"),
    ("You see, but you do not observe.", "observe", "A Scandal in Bohemia"),
    ("When you have eliminated the impossible, whatever remains must be the truth.", "eliminated", "The Sign of the Four"),
    ("Mr. Holmes, they were the footprints of a gigantic hound.", "gigantic", "The Hound of the Baskervilles"),
    ("The Time Traveller vanished three years ago.", "vanished", "The Time Machine"),
    ("No one would have believed in the last years of the nineteenth century.", "believed", "The War of the Worlds"),
    ("I beheld a transparent figure.", "transparent", "The Invisible Man"),
    ("My imagination refused to see any probable connection.", "refused", "The Island of Doctor Moreau"),
    ("He struggled with the shadow of his own failure.", "struggled", "Lord Jim"),
    ("The horror! The horror!", "horror", "Heart of Darkness"),
    ("We live as we dream, alone.", "dream", "Heart of Darkness"),
    ("Each time you happen to me all over again.", "happen", "The Age of Innocence"),
    ("The real loneliness is living among all these kind people who only ask one to pretend.", "loneliness", "The Age of Innocence"),
    ("She had learned the value of silence.", "learned", "The House of Mirth"),
    ("Guess he's been in Starkfield too many winters.", "winters", "Ethan Frome"),
    ("Life is easy to chronicle, but bewildering to practice.", "bewildering", "A Room with a View"),
    ("Only connect.", "connect", "Howards End"),
    ("Adventures do occur, but not punctually.", "occur", "A Passage to India"),
    ("Mistrust all enterprises that require new clothes.", "Mistrust", "A Room with a View"),
    ("History is a nightmare from which I am trying to awake.", "trying", "Ulysses"),
    ("Love loves to love love.", "loves", "Ulysses"),
    ("I will not serve that in which I no longer believe.", "serve", "A Portrait of the Artist as a Young Man"),
    ("His soul swooned slowly as he heard the snow falling faintly.", "falling", "Dubliners"),
    ("Mrs Dalloway said she would buy the flowers herself.", "buy", "Mrs Dalloway"),
    ("What a lark! What a plunge!", "plunge", "Mrs Dalloway"),
    ("For now she need not think about anybody.", "think", "To the Lighthouse"),
    ("So we beat on, boats against the current.", "beat", "The Great Gatsby"),
    ("I was within and without, simultaneously enchanted and repelled.", "enchanted", "The Great Gatsby"),
    ("Can't repeat the past? Why of course you can!", "repeat", "The Great Gatsby"),
    ("Her voice is full of money.", "full", "The Great Gatsby"),
    ("Isn't it pretty to think so?", "pretty", "The Sun Also Rises"),
    ("You can't get away from yourself by moving from one place to another.", "moving", "The Sun Also Rises"),
    ("The world breaks everyone and afterward many are strong at the broken places.", "breaks", "A Farewell to Arms"),
    ("I was always embarrassed by the words sacred, glorious, and sacrifice.", "embarrassed", "A Farewell to Arms"),
    ("Gregor Samsa awoke one morning from uneasy dreams.", "awoke", "The Metamorphosis"),
    ("He found himself changed in his bed into a monstrous insect.", "changed", "The Metamorphosis"),
    ("Someone must have been telling lies about Josef K.", "telling", "The Trial"),
    ("Buck did not read the newspapers.", "read", "The Call of the Wild"),
    ("He was beaten but he was not broken.", "broken", "The Call of the Wild"),
    ("Dark spruce forest frowned on either side of the frozen waterway.", "frowned", "White Fang"),
    ("The strength of the Pack is the Wolf.", "strength", "The Jungle Book"),
    ("He sat, in defiance of municipal orders, astride the gun Zam-Zammah.", "defiance", "Kim"),
    ("Peter, who was very naughty, ran straight away to Mr. McGregor's garden.", "naughty", "The Tale of Peter Rabbit"),
]

pipeline_kwargs = {
    "t5_model_name": t5_model,
    "preserve_morphology": True,
    "generation_mode": generation_mode,
    "enable_reranker": enable_reranker,
    "nli_model_name": nli_model,
    "nli_contradiction_threshold": nli_contradiction_threshold,
}

if substitute_model.exists():
    pipeline_kwargs.update(
        {
            "substitute_classifier_model_name": str(substitute_model),
            "substitute_positive_label": "valid",
        }
    )

if ranker_config.exists():
    pipeline_kwargs["ranker_config_path"] = str(ranker_config)

if learned_ranker_model.exists():
    pipeline_kwargs["learned_ranker_model_path"] = str(learned_ranker_model)

if swords_path.exists():
    pipeline_kwargs["swords_path"] = str(swords_path)
else:
    pipeline_kwargs["swords_path"] = None

examples = examples[:max_examples]

pipeline = LexicalSubstitutionPipeline(**pipeline_kwargs)
records = []

for index, (sentence, target, source) in enumerate(examples, start=1):
    print(f"[{index:03d}/{len(examples)}] {target} :: {source}", flush=True)
    record = {
        "index": index,
        "source": source,
        "sentence": sentence,
        "target": target,
        "outputs": [],
    }

    try:
        results = pipeline.substitute(sentence, target, top_k=top_k)
        record["outputs"] = [
            {
                "rank": rank,
                "word": candidate.word,
                "lexical": candidate.lexical_score,
                "target": candidate.target_score,
                "semantic": candidate.semantic_score,
                "mlm_rank": candidate.mlm_score,
                "substitute": candidate.substitute_score,
                "rerank": candidate.rerank_score,
                "final": candidate.final_score,
            }
            for rank, candidate in enumerate(results, start=1)
        ]
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"

    records.append(record)

payload = {
    "top_k": top_k,
    "model_config": {
        "t5_model": t5_model,
        "generation_mode": generation_mode,
        "enable_reranker": enable_reranker,
        "nli_model": nli_model,
        "nli_contradiction_threshold": nli_contradiction_threshold,
        "substitute_model": str(substitute_model) if substitute_model.exists() else None,
        "ranker_config": str(ranker_config) if ranker_config.exists() else None,
        "learned_ranker_model": (
            str(learned_ranker_model) if learned_ranker_model.exists() else None
        ),
        "swords_path": str(swords_path) if swords_path.exists() else None,
    },
    "examples": records,
}

output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Saved {len(records)} examples to {output_path}")
PY
