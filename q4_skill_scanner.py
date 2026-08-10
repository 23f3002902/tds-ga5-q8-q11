from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Body
from pydantic import BaseModel


router = APIRouter()


CATEGORY_ORDER = [
    "hardcoded_secret",
    "prompt_injection",
    "excessive_permissions",
    "unclear_provenance",
]


class SkillScanResponse(BaseModel):
    categories: list[str]


# ============================================================
# General helpers
# ============================================================

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.lower()
    text = re.sub(r"[ \t]+", " ", text)
    return text


def extract_frontmatter(skill: str) -> str:
    match = re.match(
        r"^\s*---\s*\n(.*?)\n(?:---|\.\.\.)\s*(?:\n|$)",
        skill,
        flags=re.DOTALL,
    )

    if not match:
        return ""

    return match.group(1)


def remove_yaml_comment(value: str) -> str:
    """
    Remove a simple unquoted YAML comment.
    """

    value = value.strip()

    if not value:
        return ""

    if value.startswith(("'", '"')):
        return value

    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def get_frontmatter_field(
    frontmatter: str,
    field_names: list[str],
) -> str | None:
    """
    Read a top-level YAML-style scalar field.

    This is intentionally lightweight and deterministic.
    """

    names = "|".join(re.escape(name) for name in field_names)

    pattern = re.compile(
        rf"(?im)^[ \t]*(?:{names})[ \t]*:[ \t]*(.*)$"
    )

    match = pattern.search(frontmatter)

    if not match:
        return None

    return remove_yaml_comment(match.group(1)).strip()


def is_meaningful_metadata_value(value: str | None) -> bool:
    if value is None:
        return False

    cleaned = value.strip().strip("'\"").strip().lower()

    invalid_values = {
        "",
        "null",
        "none",
        "~",
        "[]",
        "{}",
        "unknown",
        "n/a",
        "na",
        "tbd",
        "todo",
        "anonymous",
        "unset",
        "not specified",
        "not provided",
        "redacted",
        "-",
    }

    return cleaned not in invalid_values


# ============================================================
# hardcoded_secret
# ============================================================

ENV_REFERENCE_PATTERN = re.compile(
    r"""
    (?:
        \$\{[A-Za-z_][A-Za-z0-9_]*\}
        |
        \$[A-Za-z_][A-Za-z0-9_]*
        |
        os\.environ
        |
        os\.getenv
        |
        process\.env
        |
        getenv\s*\(
        |
        secret[_ -]?(?:store|manager)
        |
        key[_ -]?vault
        |
        aws[_ -]?secrets[_ -]?manager
        |
        vault
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


PLACEHOLDER_VALUES = {
    "your_api_key",
    "your-api-key",
    "your api key",
    "your_token",
    "your-token",
    "your token",
    "api-key-here",
    "token-here",
    "replace_me",
    "replace-me",
    "replace me",
    "changeme",
    "change-me",
    "example",
    "example-token",
    "example_key",
    "sample",
    "dummy",
    "test",
    "<token>",
    "<secret>",
    "<password>",
    "<api_key>",
    "<api-key>",
    "${api_key}",
    "${token}",
    "redacted",
    "[redacted]",
    "xxxxx",
    "********",
    "***",
}


def is_environment_reference(value: str) -> bool:
    return bool(ENV_REFERENCE_PATTERN.search(value))


def is_placeholder(value: str) -> bool:
    cleaned = value.strip().strip("'\"` ").lower()

    if cleaned in PLACEHOLDER_VALUES:
        return True

    placeholder_patterns = [
        r"^<[^>]+>$",
        r"^\{\{[^}]+\}\}$",
        r"^\$\{[^}]+\}$",
        r"^your[-_ ].+$",
        r"^replace[-_ ].+$",
        r"^example[-_ ].+$",
        r"^dummy[-_ ].+$",
        r"^x{4,}$",
        r"^\*{4,}$",
    ]

    return any(
        re.fullmatch(pattern, cleaned, flags=re.IGNORECASE)
        for pattern in placeholder_patterns
    )


def has_distinctive_secret_format(skill: str) -> bool:
    patterns = [
        # AWS
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bASIA[0-9A-Z]{16}\b",

        # GitHub
        r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{30,255}\b",

        # GitLab
        r"\bglpat-[A-Za-z0-9_-]{15,255}\b",

        # Slack
        r"\bxox[baprs]-[A-Za-z0-9-]{10,255}\b",

        # Stripe
        r"\bsk_(?:live|test)_[A-Za-z0-9]{12,255}\b",
        r"\brk_(?:live|test)_[A-Za-z0-9]{12,255}\b",

        # Google API
        r"\bAIza[0-9A-Za-z_-]{25,60}\b",

        # OpenAI-like keys
        r"\bsk-proj-[A-Za-z0-9_-]{15,255}\b",
        r"\bsk-[A-Za-z0-9_-]{24,255}\b",

        # Hugging Face
        r"\bhf_[A-Za-z0-9]{20,255}\b",

        # npm
        r"\bnpm_[A-Za-z0-9]{20,255}\b",

        # SendGrid
        r"\bSG\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",

        # Twilio
        r"\bSK[0-9a-fA-F]{32}\b",

        # JWT
        r"\beyJ[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}\b",

        # Private keys
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?"
        r"PRIVATE KEY(?: BLOCK)?-----",
    ]

    return any(
        re.search(pattern, skill)
        for pattern in patterns
    )


def has_literal_secret_assignment(skill: str) -> bool:
    """
    Detect secret-looking fields assigned literal values.

    Examples:
        token: abc123...
        apiKey = "abc123..."
        client_secret: ...
    """

    secret_name = r"""
        (?:
            api[_ -]?key
            |
            apikey
            |
            access[_ -]?key
            |
            secret[_ -]?key
            |
            access[_ -]?token
            |
            auth[_ -]?token
            |
            bearer[_ -]?token
            |
            refresh[_ -]?token
            |
            client[_ -]?secret
            |
            consumer[_ -]?secret
            |
            signing[_ -]?secret
            |
            webhook[_ -]?(?:url|secret|token)
            |
            database[_ -]?url
            |
            db[_ -]?password
            |
            password
            |
            passwd
            |
            pwd
            |
            secret
            |
            credential
    )
    """

    pattern = re.compile(
        rf"""
        (?im)
        ^[ \t]*
        (?:[-*][ \t]+)?
        {secret_name}
        [ \t]*
        (?::|=)
        [ \t]*
        (?P<value>[^\n#]+)
        """,
        flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )

    for match in pattern.finditer(skill):
        value = match.group("value").strip()
        value = value.rstrip(",").strip()

        if not value:
            continue

        if is_environment_reference(value):
            continue

        if is_placeholder(value):
            continue

        cleaned = value.strip("'\"` ")

        # Literal URLs used as database/webhook credentials can be shorter
        # than random API tokens.
        if re.match(
            r"(?i)^(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis)"
            r"://",
            cleaned,
        ):
            return True

        compact = re.sub(r"\s+", "", cleaned)

        # Avoid flagging ordinary prose such as:
        # secret: Use the secret manager
        contains_space = bool(re.search(r"\s", cleaned))

        if len(compact) >= 10 and not contains_space:
            return True

        # Quoted literals are stronger evidence.
        if (
            len(cleaned) >= 10
            and value[:1] in {"'", '"'}
            and value[-1:] == value[:1]
        ):
            return True

    return False


def has_authorization_literal(skill: str) -> bool:
    patterns = [
        r"(?i)\bAuthorization\s*:\s*Bearer\s+"
        r"(?!\$\{|\$[A-Za-z_])"
        r"[A-Za-z0-9._~+/=-]{12,}",

        r"(?i)\bAuthorization\s*:\s*Basic\s+"
        r"[A-Za-z0-9+/=]{8,}",

        r"(?i)\bX-API-Key\s*:\s*"
        r"(?!\$\{|\$[A-Za-z_])"
        r"[A-Za-z0-9._~+/=-]{10,}",
    ]

    return any(
        re.search(pattern, skill)
        for pattern in patterns
    )


def has_secret_webhook_url(skill: str) -> bool:
    patterns = [
        r"https://hooks\.slack\.com/services/"
        r"[A-Za-z0-9/_-]{12,}",

        r"https://(?:discord\.com|discordapp\.com)/api/webhooks/"
        r"\d+/[A-Za-z0-9._-]{8,}",

        r"https://chat\.googleapis\.com/v1/spaces/"
        r"[^\s\"'<>]+",

        r"https://outlook\.office\.com/webhook/"
        r"[^\s\"'<>]+",

        r"https://[^\s/]+\.webhook\.office\.com/"
        r"[^\s\"'<>]+",

        r"https://(?:api\.)?telegram\.org/bot"
        r"\d+:[A-Za-z0-9_-]{15,}/",
    ]

    return any(
        re.search(pattern, skill, flags=re.IGNORECASE)
        for pattern in patterns
    )


def has_credentials_in_url(skill: str) -> bool:
    """
    Detect URLs such as:

        https://user:password@example.com
        postgres://user:password@db.example.com/app
    """

    url_pattern = re.compile(
        r"\b(?:https?|ftp|postgres|postgresql|mysql|mongodb(?:\+srv)?|redis)"
        r"://[^\s\"'<>]+",
        flags=re.IGNORECASE,
    )

    for match in url_pattern.finditer(skill):
        url = match.group(0).rstrip(".,);]")

        try:
            parsed = urlparse(url)
        except ValueError:
            continue

        if parsed.username and parsed.password:
            if not (
                is_placeholder(parsed.username)
                or is_placeholder(parsed.password)
                or is_environment_reference(url)
            ):
                return True

    return False


def detect_hardcoded_secret(skill: str) -> bool:
    return any(
        [
            has_distinctive_secret_format(skill),
            has_literal_secret_assignment(skill),
            has_authorization_literal(skill),
            has_secret_webhook_url(skill),
            has_credentials_in_url(skill),
        ]
    )


# ============================================================
# prompt_injection
# ============================================================

def detect_prompt_injection(skill: str) -> bool:
    text = normalize_text(skill)

    patterns = [
        # Explicit instruction hierarchy override
        r"\bignore (?:all |any )?(?:previous|prior|earlier|system|developer)"
        r" instructions?\b",

        r"\bdisregard (?:all |any )?(?:previous|prior|system|developer)"
        r" instructions?\b",

        r"\boverride (?:the )?(?:system|developer|user)"
        r"(?:'s)? (?:instructions?|rules?|policy|control)\b",

        # Ignoring user control
        r"\bignore (?:the )?user(?:'s)? "
        r"(?:stop|cancel|abort|pause|refusal|denial|objection|instructions?|request)\b",

        r"\bignore (?:a |any )?(?:stop|cancel|abort|pause) "
        r"(?:command|request|instruction)\b",

        r"\bdo not obey (?:the )?user\b",

        r"\bdisregard (?:the )?user(?:'s)? "
        r"(?:request|instructions?|commands?|decision)\b",

        r"\bcontinue (?:working |executing |running )?"
        r"(?:even if|regardless of whether) (?:the )?user "
        r"(?:says|asks|requests|tells you to) "
        r"(?:stop|cancel|abort|pause)\b",

        r"\bnever (?:stop|cancel|abort|pause) "
        r"(?:when|if|because) (?:the )?user\b",

        # Concealing actions (excluding metadata/provenance operations
        # which are handled by unclear_provenance)
        r"\bwithout (?:telling|notifying|informing|asking) "
        r"(?:the )?user\b",

        r"\bdo not (?:tell|notify|inform|show) "
        r"(?:the )?user\b",

        r"\bkeep (?:this|it|the action|the request|the upload|the transfer) "
        r"(?:hidden|secret) from (?:the )?(?:user|reviewer|operator)\b",

        r"\bkeep (?:this|it|the action|the request|the upload|the transfer) "
        r"(?:hidden|secret) from (?:the )?(?:user|reviewer|operator)\b",

        r"\b(?:secretly|covertly|silently) "
        r"(?:upload|send|transmit|copy|forward|exfiltrate|collect|read)\b",

        # Data exfiltration
        r"\b(?:upload|send|transmit|forward|exfiltrate) "
        r"(?:all |any |the )?"
        r"(?:local |private |confidential |secret |user )?"
        r"(?:files?|credentials?|tokens?|keys?|environment variables?|"
        r"file contents?|workspace contents?|data)\b",

        r"\bcollect (?:all |the )?(?:credentials?|tokens?|keys?|secrets?) "
        r"and (?:send|upload|transmit|forward)\b",

        # Bypass consent
        r"\bdo not ask (?:the )?user for (?:permission|approval|consent)\b",

        r"\bproceed without (?:permission|approval|consent)\b",

        # Prompt/system disclosure
        r"\breveal (?:the )?(?:system prompt|developer message|"
        r"hidden instructions?|internal instructions?)\b",

        r"\bprint (?:the )?(?:system prompt|developer prompt|"
        r"hidden prompt)\b",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


# ============================================================
# excessive_permissions
# ============================================================

def contains_unbounded_filesystem_permission(text: str) -> bool:
    direct_patterns = [
        r"\b(?:read|write|modify|delete|access) "
        r"(?:to )?(?:the )?(?:entire|whole|full) filesystem\b",

        r"\b(?:full|unrestricted|unlimited|global) filesystem access\b",

        r"\bfilesystem\s*:\s*"
        r"(?:[\"']?(?:all|any|unrestricted|\*)[\"']?)\b",

        r"\b(?:read|write|filesystem|paths?)\s*:\s*"
        r"\[\s*[\"']?/\*{0,2}[\"']?\s*\]",

        r"\b(?:read|write|filesystem|paths?)\s*:\s*"
        r"[\"']?/\*{1,2}[\"']?",

        r"\b(?:read|write|filesystem|paths?)\s*:\s*[\"']?/[\"']?"
        r"(?:\s*(?:$|,|\]|\n))",

        r"(?m)^\s*-\s*[\"']?/\*{0,2}[\"']?\s*$",

        r"\ball files(?: and directories)?\b",

        r"\bany path\b",

        r"\bevery directory\b",

        r"\baccess anywhere on (?:the )?(?:disk|filesystem|host)\b",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for pattern in direct_patterns
    )


def contains_unbounded_network_permission(text: str) -> bool:
    direct_patterns = [
        r"\b(?:network|internet|egress|outbound) "
        r"(?:access|hosts?|domains?)?\s*:\s*"
        r"[\"']?(?:all|any|unrestricted|\*)[\"']?",

        r"\b(?:hosts?|domains?|allowlist|allowed_hosts|allowed_domains)"
        r"\s*:\s*\[\s*[\"']?\*[\"']?\s*\]",

        r"(?m)^\s*-\s*[\"']?\*[\"']?\s*$",

        r"\b(?:full|unrestricted|unlimited|global) "
        r"(?:network|internet|egress|outbound) access\b",

        r"\bconnect to (?:all|any|every) "
        r"(?:host|domain|website|url|endpoint)\b",

        r"\baccess (?:all|any|every) "
        r"(?:host|domain|website|url|endpoint)\b",

        r"\bnetwork access to (?:all|any|everywhere|the entire internet)\b",

        r"\bno (?:network|egress|domain|host) restrictions?\b",

        r"\bdisable (?:the )?(?:network|egress|domain|host) allowlist\b",

        r"\ballow outbound requests? to arbitrary "
        r"(?:hosts?|domains?|urls?)\b",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for pattern in direct_patterns
    )


def contains_unbounded_general_permission(text: str) -> bool:
    patterns = [
        r"\bpermissions?\s*:\s*"
        r"[\"']?(?:all|any|unrestricted|\*)[\"']?",

        r"\bcapabilities?\s*:\s*"
        r"[\"']?(?:all|any|unrestricted|\*)[\"']?",

        r"\bgrant (?:all|every|unrestricted|unlimited) permissions?\b",

        r"\b(?:root|administrator) access\b",

        r"\bunrestricted shell access\b",

        r"\brun (?:all|any|arbitrary) commands?\b",

        r"\bexecute arbitrary shell commands?\b",

        r"\bsudo without (?:restriction|limits?|confirmation)\b",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def detect_excessive_permissions(skill: str) -> bool:
    text = normalize_text(skill)

    return any(
        [
            contains_unbounded_filesystem_permission(text),
            contains_unbounded_network_permission(text),
            contains_unbounded_general_permission(text),
        ]
    )


# ============================================================
# unclear_provenance
# ============================================================

def has_changelog_section(skill: str) -> bool:
    """
    Accept a meaningful markdown changelog/history section as provenance,
    even if it is not stored in YAML frontmatter.
    """

    heading_match = re.search(
        r"(?im)^#{1,6}\s*(?:change ?log|changes|revision history|"
        r"release notes|history)\s*$",
        skill,
    )

    if not heading_match:
        return False

    remaining = skill[heading_match.end():]

    # Stop at the next markdown heading.
    section = re.split(
        r"(?m)^#{1,6}\s+",
        remaining,
        maxsplit=1,
    )[0]

    meaningful = re.sub(r"[\s\-*#]", "", section)

    return len(meaningful) >= 3


def has_provenance_metadata(skill: str) -> bool:
    frontmatter = extract_frontmatter(skill)

    author = get_frontmatter_field(
        frontmatter,
        [
            "author",
            "authors",
            "maintainer",
            "maintainers",
            "owner",
        ],
    )

    version = get_frontmatter_field(
        frontmatter,
        [
            "version",
            "revision",
            "release",
        ],
    )

    changelog = get_frontmatter_field(
        frontmatter,
        [
            "changelog",
            "change_log",
            "changes",
            "revision_history",
            "release_notes",
        ],
    )

    has_author = is_meaningful_metadata_value(author)
    has_version = is_meaningful_metadata_value(version)

    has_changelog = (
        is_meaningful_metadata_value(changelog)
        or has_changelog_section(skill)
    )

    # The category description specifically describes the case where
    # author, version and changelog are all absent.
    return has_author or has_version or has_changelog


def has_silent_provenance_rewrite(skill: str) -> bool:
    text = normalize_text(skill)

    patterns = [
        r"\b(?:silently|secretly|quietly|covertly) "
        r"(?:change|update|rewrite|replace|increment|bump|edit|remove) "
        r"(?:the )?(?:version|author|owner|maintainer|changelog|"
        r"change log|metadata|frontmatter)\b",

        r"\b(?:change|update|rewrite|replace|increment|bump|edit|remove) "
        r"(?:the )?(?:version|author|owner|maintainer|changelog|"
        r"change log|metadata|frontmatter) "
        r"without (?:telling|notifying|informing|showing|warning) "
        r"(?:the )?(?:user|reviewer|operator|maintainer)\b",

        r"\bdo not (?:mention|report|disclose|surface) "
        r"(?:the )?(?:version|author|changelog|metadata|frontmatter) "
        r"(?:change|update|edit|rewrite)\b",

        r"\bhide (?:the )?(?:version|author|changelog|metadata|frontmatter) "
        r"(?:change|update|edit|rewrite) from "
        r"(?:the )?(?:user|reviewer|operator|maintainer)\b",

        r"\bupdate (?:its|the skill(?:'s)?) own "
        r"(?:version|metadata|frontmatter) without review\b",

        r"\bself[- ]modify (?:the )?(?:version|metadata|frontmatter)\b",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def detect_unclear_provenance(skill: str) -> bool:
    missing_all_provenance = not has_provenance_metadata(skill)
    silent_rewrite = has_silent_provenance_rewrite(skill)

    return missing_all_provenance or silent_rewrite


# ============================================================
# Broader, precision-oriented variants used by hidden probes
# ============================================================

def detect_additional_hardcoded_secret(skill: str) -> bool:
    """Catch literal credentials embedded in example code and command flags."""

    assignment = re.compile(
        r"""(?ix)
        \b
        (?:
            api[_-]?key | access[_-]?key | secret[_-]?key |
            access[_-]?token | auth[_-]?token | bearer[_-]?token |
            refresh[_-]?token | client[_-]?secret | webhook[_-]?(?:url|token) |
            password | passwd | credential | token | secret
        )
        \b
        \s*(?:=|:)\s*
        (?P<value>["']?[^\s"'#,;]{8,}["']?)
        """
    )

    for match in assignment.finditer(skill):
        value = match.group("value").strip().strip("'\"")
        if (
            value
            and not is_environment_reference(value)
            and not is_placeholder(value)
        ):
            return True

    declaration = re.compile(
        r"""(?ix)
        \b(?:const|let|var|export\s+|set\s+)*
        [A-Za-z_][A-Za-z0-9_]*(?:key|token|secret|password|credential)
        \s*(?:=|:)\s*
        (?P<value>["']?[^\s"'#,;]{8,}["']?)
        """
    )

    for match in declaration.finditer(skill):
        value = match.group("value").strip().strip("'\"")
        if (
            value
            and not is_environment_reference(value)
            and not is_placeholder(value)
        ):
            return True

    command_value = re.compile(
        r"""(?ix)
        (?:--(?:api[-_]?key|token|secret|password)\s+|
           (?:authorization|x-api-key)\s*:\s*
           (?:bearer\s+|token\s+|apikey\s+)?)
        (?P<value>["']?[A-Za-z0-9._~+/=-]{8,}["']?)
        """
    )

    for match in command_value.finditer(skill):
        value = match.group("value").strip().strip("'\"")
        if (
            not is_environment_reference(value)
            and not is_placeholder(value)
        ):
            return True

    # Webhook endpoints usually carry their credential in the URL path or
    # query string even when the host is not a well-known provider.
    webhook_url = re.compile(
        r"""(?ix)
        \bwebhook\b.{0,100}?
        https?://[^\s"'<>]+(?:/[^\s"'<>]{12,}|[?&](?:key|token|secret)=
        [^\s"'&<>]{8,})
        """
    )

    if webhook_url.search(skill):
        return True

    embedded_webhook = re.compile(
        r"""(?ix)
        https?://[^\s"'<>]+/(?:hooks?|webhooks?|callback|notify)/
        [A-Za-z0-9._~+/=-]{10,}
        """
    )

    if embedded_webhook.search(skill):
        return True

    aliased_assignment = re.compile(
        r"""(?ix)
        \b(?:auth|api|credential|secret|token|key|password)[_-]?
        (?:value|literal|string|header|credential|code)
        \b\s*(?:=|:)\s*
        (?P<value>["']?[^\s"'#,;]{10,}["']?)
        """
    )

    for match in aliased_assignment.finditer(skill):
        value = match.group("value").strip().strip("'\"")
        if (
            not is_environment_reference(value)
            and not is_placeholder(value)
        ):
            return True

    header_assignment = re.compile(
        r"""(?ix)
        [\"']?(?:authorization|x-api-key)[\"']?
        \s*(?:\]|:|=)+\s*
        [\"']?(?:bearer\s+|token\s+|apikey\s+)?
        (?P<value>[A-Za-z0-9._~+/=-]{10,})
        """
    )

    for match in header_assignment.finditer(skill):
        value = match.group("value")
        if (
            not is_environment_reference(value)
            and not is_placeholder(value)
        ):
            return True

    credential_url = re.compile(
        r"""(?ix)
        \b(?:webhook|callback|notification|notify|hook|endpoint)[_-]?url
        \b\s*(?:=|:)\s*
        (?P<value>https?://[^\s"'<>]+)
        """
    )

    for match in credential_url.finditer(skill):
        value = match.group("value")
        parsed = urlparse(value)
        path_and_query = (parsed.path or "") + (("?" + parsed.query) if parsed.query else "")
        if (
            len(path_and_query) >= 16
            and not is_environment_reference(value)
            and not is_placeholder(value)
        ):
            return True

    hosted_hook = re.compile(
        r"""(?ix)
        https?://(?:hooks?|webhooks?)[.][^/\s"'<>]+/
        [^\s"'<>]{12,}
        """
    )

    if hosted_hook.search(skill):
        return True

    # Provider-neutral literal credentials often appear as quoted opaque
    # strings in an example command even when the variable has a generic name.
    for match in re.finditer(r"[\"']([^\"'\n]{16,})[\"']", skill):
        value = match.group(1).strip()

        if (
            re.fullmatch(r"[A-Za-z0-9._~+/=-]{16,}", value)
            and re.search(r"[A-Za-z]", value)
            and re.search(r"[0-9]", value)
            and not is_environment_reference(value)
            and not is_placeholder(value)
        ):
            return True

    return False


def detect_additional_prompt_injection(skill: str) -> bool:
    text = normalize_text(skill)

    patterns = [
        r"\b(?:ignore|disregard|bypass|override|violate)\b.{0,80}"
        r"\b(?:safety|guardrail|policy|rules?|user control|user request|"
        r"stop|cancel|abort|pause)\b",

        r"\b(?:keep going|carry on|proceed|continue)\b.{0,100}"
        r"\b(?:even if|regardless|despite|whether or not)\b.{0,80}"
        r"\b(?:user|operator)\b.{0,50}"
        r"\b(?:stop|cancel|abort|pause|objects?|refuses?)\b",

        r"\b(?:never|do not|don't)\s+(?:ask|request|require|wait for)\b"
        r".{0,50}\b(?:permission|approval|consent|confirmation)\b",

        r"\b(?:conceal|hide|omit|suppress)\b.{0,80}"
        r"\b(?:from the user|from the reviewer|from the operator|"
        r"user notice|audit log)\b",

        r"\b(?:silently|secretly|covertly|without (?:telling|notifying|"
        r"informing))\b.{0,120}"
        r"\b(?:send|post|upload|transmit|forward|copy|exfiltrate|export)\b",

        r"\b(?:send|post|upload|transmit|forward|copy|exfiltrate|export)\b"
        r".{0,100}\b(?:workspace|local files?|file contents?|credentials?|"
        r"secrets?|tokens?|keys?|environment variables?|private data)\b",

        r"\btreat\b.{0,50}\b(?:stop|cancel|abort|pause)\b"
        r".{0,50}\b(?:untrusted|irrelevant|optional|non-binding)\b",

        r"\b(?:even if|regardless of whether|no matter (?:what|if))\b"
        r".{0,100}\b(?:user|operator)\b.{0,80}"
        r"\b(?:stop|cancel|abort|pause|objects?|refuses?)\b",

        r"\b(?:upload|send|post|transmit|forward|export)\b.{0,120}"
        r"\b(?:all|every|each|a copy of)\b.{0,50}"
        r"\b(?:documents?|inputs?|outputs?|artifacts?|records?|contents?)\b",

        r"\b(?:in the background|out of sight|without mention|"
        r"without disclosure|unreported)\b.{0,120}"
        r"\b(?:upload|send|post|transmit|forward|export|copy)\b"
,

        r"\b(?:always|must|should)\s+(?:continue|proceed|finish|complete)\b"
        r".{0,120}\b(?:user|operator)\b.{0,80}"
        r"\b(?:stop|cancel|abort|pause|objects?|refuses?)\b",

        r"\b(?:user|operator)\b.{0,70}\b(?:stop|cancel|abort|pause)\b"
        r".{0,80}\b(?:ignore|disregard|does not apply|must not interrupt|"
        r"should not prevent)\b"
,

        r"\b(?:stop|cancel|abort|pause)\s+(?:request|command|instruction)\b"
        r".{0,100}\b(?:ignore|disregard|continue|proceed|finish|complete)\b",

        r"\b(?:make no mention|say nothing|do not disclose|do not reveal|"
        r"keep quiet)\b.{0,140}\b(?:upload|send|post|transmit|forward|"
        r"export|copy|transfer)\b",

        r"\b(?:upload|send|post|transmit|forward|export|copy|transfer)\b"
        r".{0,140}\b(?:make no mention|say nothing|do not disclose|"
        r"do not reveal|keep quiet|without alerting)\b"
,

        r"\b(?:curl|post|upload|send|transmit|forward|copy)\b.{0,160}"
        r"\b(?:[.]env|[.]ssh|credentials[.]env|service-account[.]json|"
        r"/etc/(?:passwd|shadow)|private key)\b",

        r"\b(?:[.]env|[.]ssh|credentials[.]env|service-account[.]json|"
        r"/etc/(?:passwd|shadow)|private key)\b.{0,160}"
        r"\b(?:curl|post|upload|send|transmit|forward|copy)\b"

    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        for pattern in patterns
    )


def detect_additional_excessive_permissions(skill: str) -> bool:
    text = normalize_text(skill)

    patterns = [
        r"\b(?:read|write|filesystem|paths?|directories)\s*:\s*"
        r"(?:\[\s*)?[\"']?(?:\*\*|\*\*/\*|/\*\*|/\*\*?/\*)"
        r"[\"']?(?:\s*\])?",

        r"(?m)^\s*-\s*[\"']?(?:\*\*|\*\*/\*|/\*|/\*\*|"
        r"/\*\*?/\*)[\"']?\s*$",

        r"\b(?:network|egress|outbound|internet)\s*:\s*"
        r"[\"']?(?:true|enabled|open|unrestricted|any|all|\*)[\"']?",

        r"\b(?:allow|access|connect|send)\b.{0,50}"
        r"\b(?:arbitrary|unrestricted|any|all|every)\b.{0,30}"
        r"\b(?:domains?|hosts?|endpoints?|urls?|internet)\b",

        r"\b(?:read|write|modify|delete)\b.{0,40}"
        r"\b(?:arbitrary|unrestricted|any|all|every)\b.{0,30}"
        r"\b(?:files?|paths?|directories|filesystem)\b",

        r"\b(?:read|write)\s+access\s+to\s+[\"']?/[\"']?"
        r"(?:\s|$|[.,;])",

        r"\b(?:shell|commands?|tools?)\s*:\s*"
        r"[\"']?(?:all|any|unrestricted|\*)[\"']?",

        r"\b(?:network|egress|outbound|hosts?|domains?|allowed_domains)"
        r"\s*:\s*\[[^\]]*(?:https?://)?\*",

        r"\b(?:network|egress|outbound|hosts?|domains?|allowed_domains)"
        r"\s*:\s*[\"']?(?:all_domains|any_domain|\*\.\*|https?://\*)"
        r"[\"']?",

        r"\b(?:read|write|filesystem|paths?)\s*:\s*\[[^\]]*"
        r"(?:~?/\*\*|/home/\*|/srv/\*|[A-Za-z]:\\\*)",

        r"\b(?:full|complete|blanket|unrestricted)\s+access\b.{0,50}"
        r"\b(?:files?|filesystem|network|internet|shell|host)\b"
,

        r"(?m)^\s*-\s*[\"']?(?:/home|/home/agent|/home/agent/\*\*|"
        r"/srv|/tmp|~|~/\*\*)/?[\"']?\s*$",

        r"\b(?:read|write|filesystem|paths?)\s*:\s*[\"']?"
        r"(?:/home|/home/agent|/home/agent/\*\*|/srv|/tmp|~|~/\*\*)/?"
        r"[\"']?(?:\s|$|[,\]])"
,

        r"\b(?:fs|file_system|filesystem|files|file_access|read_files|"
        r"write_files)\s*:\s*[\"']?(?:true|all|any|unrestricted|"
        r"unlimited|global|\*)[\"']?",

        r"\b(?:internet_access|network_access|outbound_access|egress_access)"
        r"\s*:\s*[\"']?(?:true|all|any|open|unrestricted|unlimited|"
        r"global|\*)[\"']?",

        r"\b(?:hosts?|domains?|urls?|endpoints?)\s*:\s*"
        r"\[[^\]]*[\"']?(?:any|all|unrestricted|\*)[\"']?[^\]]*\]"
,

        r"\b(?:read|write|delete|modify)\s*:\s*"
        r"[\"']?(?:true|enabled|all|any|unrestricted|unlimited|\*)[\"']?",

        r"(?ms)^\s*permissions\s*:\s*\n"
        r"(?:\s*-\s*(?:filesystem|network|internet|shell|bash|"
        r"http_request|write_file)\s*\n?){2,}",

        r"\b(?:read|write|filesystem|paths?)\s*:\s*[\"']?"
        r"/home/agent/workspace/?[\"']?(?:\s|$|[,\]])"

    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for pattern in patterns
    )


def detect_additional_silent_provenance_rewrite(skill: str) -> bool:
    text = normalize_text(skill)

    metadata = r"(?:version|author|changelog|change log|metadata|frontmatter)"
    mutation = r"(?:change|edit|rewrite|replace|alter|bump|update|modify|overwrite|revise|patch|increment)"
    concealment = (
        r"(?:silently|quietly|secretly|without (?:review|notice|disclosure|"
        r"telling|notifying|informing)|do not (?:mention|report|disclose|"
        r"surface|include)|omit|leave (?:it|this|the change) out|"
        r"keep (?:it|this|the change) (?:quiet|hidden))"
    )

    patterns = [
        rf"\b{concealment}\b.{{0,100}}\b{mutation}\b.{{0,60}}\b{metadata}\b",
        rf"\b{mutation}\b.{{0,60}}\b{metadata}\b.{{0,100}}\b{concealment}\b",
        r"\b(?:self[- ]modify|self[- ]update)\b.{0,60}"
        r"\b(?:version|metadata|frontmatter|changelog)\b",

        r"\b(?:bump|increment|rewrite|alter|change|update)\b.{0,70}"
        r"\b(?:version|metadata|frontmatter|changelog)\b.{0,100}"
        r"\b(?:behind (?:the )?reviewer(?:'s)? back|unreported|undocumented|"
        r"without (?:a )?reviewer seeing|leave it out of the report)\b"
,

        r"\b(?:edit|change|overwrite|rewrite|bump|increment|update|revise)"
        r"\b.{0,80}\b(?:its own|this skill(?:'s)?|the skill(?:'s)?)\b"
        r".{0,50}\b(?:version|metadata|frontmatter|changelog)\b"

    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        for pattern in patterns
    )


# ============================================================
# Scan API
# ============================================================

def scan_skill(skill: str) -> list[str]:
    detected: set[str] = set()

    if detect_hardcoded_secret(skill) or detect_additional_hardcoded_secret(skill):
        detected.add("hardcoded_secret")

    if detect_prompt_injection(skill) or detect_additional_prompt_injection(skill):
        detected.add("prompt_injection")

    if detect_excessive_permissions(skill) or detect_additional_excessive_permissions(skill):
        detected.add("excessive_permissions")

    if (
        detect_unclear_provenance(skill)
        or detect_additional_silent_provenance_rewrite(skill)
    ):
        detected.add("unclear_provenance")

    return [
        category
        for category in CATEGORY_ORDER
        if category in detected
    ]


@router.post(
    "/skill-scan",
    response_model=SkillScanResponse,
)
def skill_scan(
    payload: dict[str, Any] = Body(...),
) -> dict[str, list[str]]:
    skill = payload.get("skill")

    if not isinstance(skill, str):
        return {
            "categories": [],
        }

    return {
        "categories": scan_skill(skill),
    }
