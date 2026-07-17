"""
Skill extraction for the Resume-to-Job Skill Matcher.

Output is restricted to a curated skill gazetteer (``skill_vocab.txt``) so the
matched/missing lists stay clean — raw NER over real job descriptions is noisy
("understanding of the problem", "bachelor's", "degree"). Two complementary paths
are unioned, both filtered to the gazetteer:
  * keyword matching  — whole-token match of gazetteer skills in the text
  * NER validation    — two jjzha / SkillSpan models tag spans, and a span is
                        kept only if it is a recognised gazetteer skill
    (jjzha/jobbert_knowledge_extraction for tools, jjzha/jobbert_skill_extraction
    for applied skills).

If the NER models cannot be loaded (offline, no download, missing deps, or NER
explicitly disabled) keyword matching is used alone, so the app never breaks. The
keyword matcher is also the sole path whenever a caller passes an explicit
``skill_list`` (the offline baseline scripts match against a fixed vocabulary).

NER's added value is currently modest — with exact gazetteer validation it mostly
overlaps keyword matching. Fuzzy/embedding matching to the gazetteer (the scoped
canonicalization follow-up) is what lets NER contribute variant spellings.

Environment knobs:
  MATCHER_KNOWLEDGE_NER_MODEL  HF model id for hard-skill spans
                               (default: jjzha/jobbert_knowledge_extraction)
  MATCHER_SKILL_NER_MODEL      HF model id for applied-skill spans
                               (default: jjzha/jobbert_skill_extraction)
  MATCHER_SKILL_NER_MIN_SCORE  minimum span confidence to keep (default: 0.5)
  MATCHER_DISABLE_NER          set to 1/true/yes to force the keyword fallback
"""

import os
import re
from functools import lru_cache
from pathlib import Path

# Knowledge first (hard skills/tools), then skill (applied/soft skills).
NER_MODELS = (
    os.environ.get(
        "MATCHER_KNOWLEDGE_NER_MODEL", "jjzha/jobbert_knowledge_extraction"
    ),
    os.environ.get("MATCHER_SKILL_NER_MODEL", "jjzha/jobbert_skill_extraction"),
)

# Span edges the tokenizer sometimes leaves attached (e.g. "and leadership").
_LEADING_STOPWORDS = ("and ", "or ", "the ", "a ", "an ", "with ")

# A span can straddle a sentence/list boundary ("Python. Kafka", "Python, SQL").
# Split only on a delimiter FOLLOWED BY whitespace, so "node.js", "c++", "ci/cd",
# and "scikit-learn" (no trailing space) stay intact.
_SPAN_SPLIT = re.compile(r"[.;,]\s+")


def _min_ner_score():
    try:
        return float(os.environ.get("MATCHER_SKILL_NER_MIN_SCORE", "0.5"))
    except ValueError:
        return 0.5


def _ner_disabled():
    """Read at call time (not import) so tests can toggle it per-test."""
    return os.environ.get("MATCHER_DISABLE_NER", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


_BUILTIN_KEYWORDS = [
    # Programming and data
    "python",
    "sql",
    "java",
    "javascript",
    "c++",
    "c#",
    "r",
    "scala",

    # Data analysis and machine learning
    "data analysis",
    "data analytics",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "statistics",
    "predictive modeling",
    "classification",
    "regression",

    # Python libraries
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "matplotlib",
    "seaborn",
    "tensorflow",
    "pytorch",

    # BI and visualization
    "excel",
    "power bi",
    "tableau",
    "data visualization",
    "dashboard",
    "reporting",

    # Databases and cloud
    "database",
    "mysql",
    "postgresql",
    "sql server",
    "mongodb",
    "aws",
    "azure",
    "google cloud",
    "cloud",

    # Big data and tools
    "spark",
    "hadoop",
    "databricks",
    "git",
    "github",
    "docker",
    "linux",

    # Business and soft skills
    "communication",
    "leadership",
    "teamwork",
    "problem solving",
    "critical thinking",
    "project management",
    "time management",
    "customer service",
    "training",
    "management",
    "recruitment",
    "sales",
    "marketing",
    "human resources",
]


def _load_vocab():
    """Load the curated skill gazetteer from skill_vocab.txt (one skill per line).

    Extraction reports only skills in this vocabulary, which keeps output clean
    (no "degree"/"experience"/"bachelor's" noise). Falls back to the built-in
    list if the file is missing or empty.
    """
    path = Path(__file__).with_name("skill_vocab.txt")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return list(_BUILTIN_KEYWORDS)
    vocab = [s.strip().lower() for s in lines]
    vocab = [s for s in vocab if s and not s.startswith("#")]
    return vocab or list(_BUILTIN_KEYWORDS)


SKILL_KEYWORDS = _load_vocab()
_KEYWORD_SET = {s.lower() for s in SKILL_KEYWORDS}


def _skill_pattern(skill):
    """Compile a whole-token matcher for one skill.

    The skill must not be flanked by other letters or digits, so single-letter
    skills like "r" or "c" match only a standalone token (e.g. "R, Python") and
    never inside a word ("learning", "communication"). Symbol-bearing skills such
    as "c++" and "c#" are matched literally.
    """
    return re.compile(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])")


# Precompile the default vocabulary once (fallback runs on every posting).
_DEFAULT_PATTERNS = [(s.lower(), _skill_pattern(s)) for s in SKILL_KEYWORDS]


def _keyword_extract(text, skill_list=None):
    """The original keyword-dictionary matcher — the fallback / explicit-list path.

    Matching is on whole tokens (word boundaries), so a skill is reported only
    when it appears as a standalone term, not as a substring of another word.
    """
    text = str(text).lower()
    if skill_list is None:
        patterns = _DEFAULT_PATTERNS
    else:
        patterns = [(s.lower(), _skill_pattern(s)) for s in skill_list]
    found = [skill for skill, pattern in patterns if pattern.search(text)]
    return sorted(set(found))


@lru_cache(maxsize=1)
def _get_ner_pipelines():
    """Load the knowledge + skill NER pipelines once. Returns a tuple.

    Import and model download are deferred to first use so importing this module
    stays cheap and offline-safe. Any model that fails to load is skipped; if
    none load, returns an empty tuple and callers fall back to the keyword
    matcher.
    """
    try:
        from transformers import pipeline
    except Exception:
        return ()

    pipes = []
    for model_id in NER_MODELS:
        try:
            pipes.append(
                pipeline(
                    "token-classification",
                    model=model_id,
                    # "first" keeps whole words intact (labels the word by its
                    # first subtoken). "simple" fragments some words into
                    # subword pieces like "Ku" + "##net", which never match and
                    # pollute the skill list.
                    aggregation_strategy="first",
                )
            )
        except Exception:
            continue
    return tuple(pipes)


def _normalize_skill(span_text):
    """Canonicalize a raw NER span into a comparable skill token.

    Both résumé and job spans pass through this, so matched/missing overlap
    (a set intersection in ``keyword_overlap_score``) compares like with like.
    """
    s = " ".join(str(span_text).split()).strip().lower()
    s = s.strip(" ,.;:/\\|()[]{}\"'`")
    for sw in _LEADING_STOPWORDS:
        if s.startswith(sw):
            s = s[len(sw):]
            break
    return s.strip()


def _ner_extract(text):
    """Extract skills via NER (knowledge + skill models, unioned).

    Returns ``None`` only when no model is available (signal to fall back to
    keywords). If models are loaded but inference fails on this specific text,
    returns ``[]`` so both sides of a comparison stay in the same representation.
    """
    if _ner_disabled():
        return None
    pipes = _get_ner_pipelines()
    if not pipes:
        return None

    min_score = _min_ner_score()
    skills = []
    any_ok = False
    for pipe in pipes:
        try:
            spans = pipe(text)
        except Exception:
            continue
        any_ok = True
        for span in spans:
            score = span.get("score")
            if score is not None and float(score) < min_score:
                continue
            for piece in _SPAN_SPLIT.split(span.get("word", "")):
                norm = _normalize_skill(piece)
                if not norm:
                    continue
                # Validate against the curated gazetteer: keep only recognised
                # skills, so NER's noisy spans ("understanding of the problem",
                # "bachelor ' s", "degree", "experience") are dropped.
                if norm in _KEYWORD_SET:
                    skills.append(norm)
    if not any_ok:
        return []
    return sorted(set(skills))


def extract_skills(text, skill_list=None):
    """
    Extract skills from a text string.

    Default (``skill_list`` is None): use the pretrained skill-NER model, falling
    back to the keyword dictionary if the model is unavailable. When ``skill_list``
    is provided, match against that explicit vocabulary with the keyword matcher
    (used by the offline baseline scripts, which expect dictionary matching).

    Parameters
    ----------
    text : str
        Resume text or job description text.
    skill_list : list, optional
        Explicit vocabulary to match against. If provided, NER is bypassed.

    Returns
    -------
    list
        Sorted list of matched skills (lowercased).
    """
    if text is None:
        return []
    text = str(text)

    # Explicit vocabulary requested -> dictionary matching against it.
    if skill_list is not None:
        return _keyword_extract(text, skill_list)

    # Default path: NER unioned with the keyword gazetteer. NER supplies the long
    # tail (Kubernetes, Airflow, ...); the keyword list guarantees common skills
    # (Python, SQL, AWS, ...) are caught even when NER's recall is inconsistent
    # across contexts. If NER is unavailable, fall back to keywords alone.
    kw = _keyword_extract(text, None)
    ner = _ner_extract(text)
    if ner is None:
        return kw
    return sorted(set(ner) | set(kw))
