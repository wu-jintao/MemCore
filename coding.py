"""Deterministic coding-memory features for CAMBench-style engineering histories.

The extractor is benchmark-independent. It identifies engineering vocabulary,
paths, symbols, exceptions, commands, task intent, and verification signals from
raw memory text. It never calls a model and never rewrites source evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata


_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_./-]+"
    r"|(?<![A-Za-z0-9_])[A-Za-z0-9_.-]+\.(?:py|pyi|js|jsx|ts|tsx|java|go|rs|cpp|cc|c|h|hpp|cs|rb|php|swift|kt|kts|scala|sql|sh|bash|zsh|yaml|yml|json|toml|ini|cfg|md|rst|html|css|scss|vue|svelte)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_QUALIFIED_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)+)(?![A-Za-z0-9_])"
)
_EXCEPTION_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception|Failure|Warning)|SIG[A-Z0-9]+|HTTP\s+[45][0-9]{2})(?![A-Za-z0-9_])"
)
_TEST_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:test_[A-Za-z0-9_]+|[A-Za-z0-9_]+_test|[A-Za-z0-9_.-]+::test_[A-Za-z0-9_]+)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_COMMAND_RE = re.compile(
    r"(?im)(?:^|[`:\n]\s*)(?:\$\s*)?((?:python(?:3)?|pytest|tox|nox|npm|pnpm|yarn|cargo|go|mvn|gradle|make|cmake|git|docker|kubectl|bash|sh)\b[^\n`]{0,180})"
)

_GENERIC_IDENTIFIERS = frozenset({
    "api", "http", "https", "json", "sql", "html", "css", "cli", "url", "uri",
    "id", "ids", "db", "sdk", "cpu", "gpu", "ui", "ux", "io", "os", "config",
    "error", "exception", "test", "tests", "class", "function", "method", "module",
})

_TASK_PATTERNS = {
    "bugfix": re.compile(
        r"\b(?:bug|defect|regression|broken|incorrect|wrong|crash|failure|failing|fails|fix|repair|traceback|exception|error)\b",
        re.IGNORECASE,
    ),
    "feature": re.compile(
        r"\b(?:feature|enhancement|implement|implementation|support|add|introduce|extend|enable|new behavior|new api)\b",
        re.IGNORECASE,
    ),
    "performance": re.compile(
        r"\b(?:performance|latency|throughput|slow|timeout|memory leak|optimi[sz]|cache|concurren|race condition|deadlock)\b",
        re.IGNORECASE,
    ),
    "refactor": re.compile(r"\b(?:refactor|cleanup|restructure|rename|deprecat|migration|upgrade)\b", re.IGNORECASE),
}

_STAGE_PATTERNS = {
    "root_cause": re.compile(r"\b(?:root cause|caused by|because|the issue was|diagnos|bisect|culprit)\b", re.IGNORECASE),
    "patch": re.compile(r"\b(?:patch|diff|changed|replace|modify|updated|fix(?:ed)?|implementation)\b", re.IGNORECASE),
    "verification": re.compile(r"\b(?:verified|validation|tests? pass(?:ed)?|all tests|regression test|reproduc|confirmed|assert)\b", re.IGNORECASE),
    "failure": re.compile(r"\b(?:did not work|still fail|failed attempt|unsuccessful|reverted|rollback|not fixed|remaining failure)\b", re.IGNORECASE),
}

_LANGUAGE_BY_SUFFIX = {
    "py": "python", "pyi": "python", "js": "javascript", "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript", "java": "java", "go": "go", "rs": "rust",
    "cpp": "cpp", "cc": "cpp", "c": "c", "h": "c", "hpp": "cpp", "cs": "csharp",
    "rb": "ruby", "php": "php", "swift": "swift", "kt": "kotlin", "kts": "kotlin",
    "scala": "scala", "sql": "sql", "sh": "shell", "bash": "shell", "zsh": "shell",
}


@dataclass(frozen=True)
class CodingFeatures:
    weighted: Counter[str]
    task_types: frozenset[str]
    stages: frozenset[str]
    paths: frozenset[str]
    symbols: frozenset[str]


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _add(counter: Counter[str], prefix: str, value: str, weight: int) -> None:
    normalized = _norm(value)
    if normalized:
        counter[prefix + normalized] = max(counter[prefix + normalized], weight)


def _path_variants(path: str) -> list[str]:
    normalized = _norm(path).strip("`'\"()[]{}.,:;")
    if not normalized or "://" in normalized:
        return []
    if normalized.startswith(("/home/", "/users/", "/usr/", "/opt/", "/tmp/", "/var/")):
        marker = "/site-packages/"
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
        else:
            return []
    parts = [part for part in normalized.split("/") if part and part not in (".", "..")]
    if not parts or any(part in ("github.com", "docs.python.org", "readthedocs.io") for part in parts):
        return []
    if all(part.isdigit() or re.fullmatch(r"v?\d+(?:\.\d+)*", part) for part in parts):
        return []
    # Require a file suffix or at least one identifier-like directory and basename.
    basename = parts[-1]
    has_suffix = bool(re.search(r"\.[a-z0-9]{1,8}$", basename))
    if not has_suffix and (len(parts) < 2 or not any(re.search(r"[a-z_]", part) for part in parts)):
        return []
    normalized = "/".join(parts)
    output = [normalized]
    name = normalized.rsplit("/", 1)[-1]
    if name != normalized:
        output.append(name)
    if len(parts) >= 2:
        output.append("/".join(parts[-2:]))
    return list(dict.fromkeys(output))


def _symbol_parts(symbol: str) -> list[str]:
    normalized = _norm(symbol).strip("`'\"()[]{}.,:;")
    values = [normalized]
    values.extend(part for part in re.split(r"::|\.", normalized) if len(part) > 2)
    values.extend(part for part in normalized.split("_") if len(part) > 2)
    return list(dict.fromkeys(value for value in values if value and value not in _GENERIC_IDENTIFIERS))


def extract_features(text: str) -> CodingFeatures:
    """Extract stable code-memory features without interpreting benchmark labels."""
    counter: Counter[str] = Counter()
    paths: set[str] = set()
    symbols: set[str] = set()

    for match in _PATH_RE.finditer(text):
        for value in _path_variants(match.group(0)):
            paths.add(value)
            _add(counter, "path:", value, 8 if "/" in value else 6)
        suffix = match.group(0).rsplit(".", 1)[-1].casefold()
        if suffix in _LANGUAGE_BY_SUFFIX:
            _add(counter, "lang:", _LANGUAGE_BY_SUFFIX[suffix], 3)

    for regex, base_weight in ((_QUALIFIED_RE, 7), (_TEST_RE, 7), (_EXCEPTION_RE, 8)):
        for match in regex.finditer(text):
            for value in _symbol_parts(match.group(0)):
                symbols.add(value)
                _add(counter, "symbol:", value, base_weight)

    for match in _COMMAND_RE.finditer(text):
        command = re.sub(r"\s+", " ", _norm(match.group(1)))
        _add(counter, "command:", command, 7)
        executable = command.split(" ", 1)[0]
        _add(counter, "tool:", executable, 4)

    task_types = frozenset(name for name, pattern in _TASK_PATTERNS.items() if pattern.search(text))
    stages = frozenset(name for name, pattern in _STAGE_PATTERNS.items() if pattern.search(text))
    for name in task_types:
        _add(counter, "task:", name, 3)
    for name in stages:
        # Verification/root-cause matches are more valuable than generic patch language.
        weight = {"root_cause": 5, "verification": 5, "patch": 3, "failure": 2}[name]
        _add(counter, "stage:", name, weight)

    lowered = text.casefold()
    for marker, label in (
        ("traceback (most recent call last)", "python_traceback"),
        ("assertionerror", "assertion"),
        ("segmentation fault", "segfault"),
        ("connection refused", "connection_refused"),
        ("timed out", "timeout"),
        ("timeout", "timeout"),
        ("permission denied", "permission"),
        ("not found", "not_found"),
    ):
        if marker in lowered:
            _add(counter, "failure:", label, 6)

    return CodingFeatures(counter, task_types, stages, frozenset(paths), frozenset(symbols))



def _clip(value: str, limit: int) -> str:
    value = str(value or "").strip()
    return value if len(value) <= limit else value[:limit] + "\n[clipped]"


def _diff_signal(text: str, limit: int = 6000) -> str:
    """Keep changed files, hunks, and changed lines instead of an entire patch."""
    output = []
    used = 0
    for raw in text.splitlines():
        line = raw.rstrip()
        keep = (line.startswith(("diff --git ", "--- ", "+++ ", "@@ "))
                or (line.startswith(("+", "-")) and not line.startswith(("+++", "---"))))
        if not keep:
            continue
        addition = line + "\n"
        if used + len(addition) > limit:
            break
        output.append(line)
        used += len(addition)
    return "\n".join(output)


def _labeled_section(text: str, label: str, next_labels: tuple[str, ...]) -> str:
    marker = label + ":"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = len(text)
    for next_label in next_labels:
        position = text.find("\n\n" + next_label + ":", start)
        if position >= 0:
            end = min(end, position)
    return text[start:end].strip()


def semantic_document(text: str) -> str:
    """Compact engineering representation for the Coding dense channel.

    Raw evidence remains stored and returned unchanged. Only embedding input is
    compacted, using deterministic source-derived signals.
    """
    features = extract_features(text)
    labels = ("HISTORICAL TASK", "HISTORICAL PATCH", "HISTORICAL TEST PATCH",
              "FAIL_TO_PASS", "PASS_TO_PASS")
    task = _labeled_section(text, labels[0], labels[1:]) or text
    patch = _labeled_section(text, labels[1], labels[2:])
    test_patch = _labeled_section(text, labels[2], labels[3:])
    fail_tests = _labeled_section(text, labels[3], labels[4:])
    pass_tests = _labeled_section(text, labels[4], ())
    feature_names = sorted(features.weighted, key=lambda name: (-features.weighted[name], name))[:96]
    parts = [
        "Engineering memory for reuse.",
        "Task types: " + ", ".join(sorted(features.task_types)) if features.task_types else "",
        "Outcome stages: " + ", ".join(sorted(features.stages)) if features.stages else "",
        "Engineering features: " + " ".join(feature_names) if feature_names else "",
        "Task description:\n" + _clip(task, 7000),
        "Patch signal:\n" + (_diff_signal(patch, 5000) or _diff_signal(text, 5000)),
        "Test patch signal:\n" + _diff_signal(test_patch, 3000),
        "Failing tests: " + _clip(fail_tests, 2500) if fail_tests else "",
        "Regression tests: " + _clip(pass_tests, 2500) if pass_tests else "",
    ]
    return "\n\n".join(part for part in parts if part and not part.endswith(":\n"))


def semantic_query(text: str) -> str:
    features = query_profile(text)
    feature_names = sorted(features.weighted, key=lambda name: (-features.weighted[name], name))[:96]
    parts = [
        "Software engineering task seeking reusable repository experience.",
        "Task types: " + ", ".join(sorted(features.task_types)) if features.task_types else "",
        "Engineering features: " + " ".join(feature_names) if feature_names else "",
        "Current task:\n" + _clip(text, 12000),
    ]
    return "\n\n".join(part for part in parts if part)

def quality_prior(features: CodingFeatures) -> float:
    """Small deterministic preference for reusable, verified engineering outcomes."""
    value = 0.0
    if "root_cause" in features.stages:
        value += 0.025
    if "verification" in features.stages:
        value += 0.035
    if "patch" in features.stages:
        value += 0.015
    if "failure" in features.stages and "verification" not in features.stages:
        value -= 0.025
    if features.paths:
        value += min(0.02, len(features.paths) * 0.004)
    if features.symbols:
        value += min(0.02, len(features.symbols) * 0.003)
    return value


def query_profile(text: str) -> CodingFeatures:
    return extract_features(text)


def query_confidence(features: CodingFeatures) -> float:
    """Confidence that exact engineering features should override generic BM25.

    A small, precise set of paths/exceptions/tests is strong. Huge path/symbol
    sets usually come from environment dumps or broad issue templates and are
    deliberately damped so indirect experience transfer can remain lexical/dense.
    """
    exceptions = sum(name.startswith(("symbol:", "failure:")) and
                     (name.endswith(("error", "exception", "failure")) or
                      name.startswith("failure:"))
                     for name in features.weighted)
    tests = sum("test_" in name or "::test" in name for name in features.weighted)
    commands = sum(name.startswith("command:") for name in features.weighted)
    precise = len(features.paths) + exceptions + tests + commands
    volume = len(features.paths) + len(features.symbols)
    if precise == 0:
        return 0.35 if features.task_types else 0.0
    confidence = min(1.0, 0.35 + precise * 0.12)
    if volume > 40:
        confidence *= 0.55
    elif volume > 20:
        confidence *= 0.75
    return max(0.2, confidence)
