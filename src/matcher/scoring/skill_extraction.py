"""
Skill extraction utilities for the Resume-to-Job Skill Matcher.

This file uses a simple keyword dictionary approach.
It checks whether known skills appear in resume text or job text.
"""

import re


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


def _skill_pattern(skill):
    """Compile a whole-token matcher for one skill.

    The skill must not be flanked by other letters or digits, so single-letter
    skills like "r" or "c" match only a standalone token (e.g. "R, Python") and
    never inside a word ("learning", "communication"). Symbol-bearing skills such
    as "c++" and "c#" are matched literally.
    """
    return re.compile(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])")


# Precompile the default vocabulary once (extract_skills runs on every posting).
_DEFAULT_PATTERNS = [(s.lower(), _skill_pattern(s)) for s in SKILL_KEYWORDS]


def extract_skills(text, skill_list=None):
    """
    Extract known skills from a text string.

    Matching is done on whole tokens (word boundaries), so a skill is reported
    only when it appears as a standalone term, not as a substring of another word.

    Parameters
    ----------
    text : str
        Resume text or job description text.

    skill_list : list, optional
        List of skills to search for.
        If no list is provided, SKILL_KEYWORDS is used.

    Returns
    -------
    list
        Sorted list of matched skills.
    """
    if text is None:
        return []

    text = str(text).lower()

    if skill_list is None:
        patterns = _DEFAULT_PATTERNS
    else:
        patterns = [(s.lower(), _skill_pattern(s)) for s in skill_list]

    found_skills = [skill for skill, pattern in patterns if pattern.search(text)]

    return sorted(set(found_skills))