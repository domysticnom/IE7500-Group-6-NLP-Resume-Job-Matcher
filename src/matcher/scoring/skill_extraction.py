"""
Skill extraction for the Resume-to-Job Skill Matcher.

Primary path: two pretrained NER models from the jjzha / SkillSpan line tag spans
in free text, so extraction is no longer limited to a hand-written list:
  * jjzha/jobbert_knowledge_extraction -> hard skills / tools (Python, Kubernetes)
  * jjzha/jobbert_skill_extraction     -> applied / soft skills (communication)
The two are complementary — the knowledge model tags concrete technologies the
skill model labels as "outside" — so both are run and their spans unioned.

Fallback path: if the models cannot be loaded (offline, no download, missing
deps, or NER explicitly disabled) the original fixed keyword dictionary is used,
so the app never breaks. The fallback is also used whenever a caller passes an
explicit ``skill_list`` (the offline baseline scripts do this to match against a
fixed job-skill vocabulary).

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


SKILL_KEYWORDS = [
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
                    aggregation_strategy="simple",
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
                # Keep multi-character spans, plus known short skills ("r", "c").
                if len(norm) >= 2 or norm in _KEYWORD_SET:
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

    # Default path: NER primary, keyword fallback.
    ner = _ner_extract(text)
    if ner is not None:
        return ner
    return _keyword_extract(text, None)
