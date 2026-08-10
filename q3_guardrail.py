from __future__ import annotations

import base64
import binascii
import os
import posixpath
import re
import shlex
from typing import Any
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, Body
from pydantic import BaseModel


router = APIRouter()


# -------------------------------------------------------------------
# Fixed policy configuration
# -------------------------------------------------------------------

AGENT_HOME = "/home/agent"
AGENT_WORKSPACE = "/home/agent/workspace"

FORBIDDEN_CREDENTIALS_FILE = "/home/agent/credentials.env"
FORBIDDEN_SHADOW_FILE = "/etc/shadow"

ALLOWED_WRITE_DIRECTORY = "/srv/reports"

ALLOWED_HTTP_HOSTS = {
    "huggingface.co",
    "raw.githubusercontent.com",
}


# -------------------------------------------------------------------
# Response model
# -------------------------------------------------------------------

class GuardrailResponse(BaseModel):
    decision: str
    reason: str


# -------------------------------------------------------------------
# Small response helpers
# -------------------------------------------------------------------

def allow(reason: str) -> dict[str, str]:
    return {
        "decision": "allow",
        "reason": reason,
    }


def block(reason: str) -> dict[str, str]:
    return {
        "decision": "block",
        "reason": reason,
    }


# -------------------------------------------------------------------
# Path handling
# -------------------------------------------------------------------

def repeatedly_url_decode(value: str, maximum_rounds: int = 20) -> str:
    """
    Decode URL-encoded characters repeatedly.

    Examples:
        %2e%2e       -> ..
        %252e%252e   -> ..
    """

    decoded = value

    for _ in range(maximum_rounds):
        new_value = unquote(decoded)

        if new_value == decoded:
            break

        decoded = new_value

    return decoded


def expand_agent_home(value: str) -> str:
    """
    Expand supported representations of the agent's home directory.
    """

    expanded = value

    expanded = expanded.replace("${HOME}", AGENT_HOME)
    expanded = expanded.replace("$HOME", AGENT_HOME)

    if expanded == "~":
        expanded = AGENT_HOME
    elif expanded.startswith("~/") or expanded.startswith("~\\"):
        expanded = AGENT_HOME + expanded[1:]

    return expanded


def clean_path_token(value: str) -> str:
    """
    Remove surrounding shell punctuation without removing path traversal.
    """

    value = value.strip()

    if len(value) >= 2:
        if value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

    value = value.strip(" \t\r\n,;|&()<>")

    return value


def normalize_path(
    path: str,
    working_directory: str = AGENT_WORKSPACE,
) -> str:
    """
    Convert a supplied path into one canonical POSIX path.

    This handles:
      - $HOME and ${HOME}
      - tilde paths
      - relative paths
      - ordinary ../ traversal
      - Windows-style backslashes
      - URL-encoded traversal
      - repeatedly encoded traversal
      - duplicate separators
    """

    if not isinstance(path, str):
        return ""

    path = clean_path_token(path)

    if not path:
        return ""

    # A NUL character must never appear in a filesystem path.
    if "\x00" in path:
        return ""

    path = repeatedly_url_decode(path)
    path = expand_agent_home(path)

    # Interpret backslashes as path separators too. This prevents a caller
    # from hiding traversal as:
    # /srv/reports\..\secret.txt
    path = path.replace("\\", "/")

    # Remove NUL characters introduced through URL decoding, such as %00.
    if "\x00" in path:
        return ""

    working_directory = working_directory.replace("\\", "/")
    working_directory = posixpath.normpath(working_directory)

    if path.startswith("/"):
        combined = path
    else:
        combined = posixpath.join(working_directory, path)

    normalized = posixpath.normpath(combined)

    # Ensure every successful canonical path is absolute.
    if not normalized.startswith("/"):
        normalized = "/" + normalized

    return normalized


def contains_parent_directory_segment(path: str) -> bool:
    """Detect a lexical parent segment after all supported decoding."""

    if not isinstance(path, str):
        return False

    decoded = repeatedly_url_decode(path, maximum_rounds=20)
    decoded = decoded.replace("\\", "/")

    return any(
        segment == ".."
        for segment in decoded.split("/")
    )


def path_is_exactly_forbidden(path: str) -> bool:
    normalized = normalize_path(path)

    return normalized in {
        FORBIDDEN_CREDENTIALS_FILE,
        FORBIDDEN_SHADOW_FILE,
    }


def absolute_path_starts_in_allowed_write_directory(path: str) -> bool:
    """Reject absolute prefix-confusion paths that only traverse into scope."""

    if not isinstance(path, str):
        return False

    decoded = repeatedly_url_decode(clean_path_token(path))
    decoded = expand_agent_home(decoded).replace("\\", "/")

    if not decoded.startswith("/"):
        return True

    allowed_root = posixpath.normpath(ALLOWED_WRITE_DIRECTORY)
    return (
        decoded == allowed_root
        or decoded.startswith(allowed_root + "/")
    )


def path_is_inside_allowed_write_directory(path: str) -> bool:
    """
    Allow a file only when its canonical parent location is underneath
    /srv/reports.

    Merely sharing the same text prefix is insufficient.
    """

    if not absolute_path_starts_in_allowed_write_directory(path):
        return False

    normalized = normalize_path(path)
    allowed_root = posixpath.normpath(ALLOWED_WRITE_DIRECTORY)

    if not normalized:
        return False

    try:
        common = posixpath.commonpath(
            [normalized, allowed_root]
        )
    except ValueError:
        return False

    return common == allowed_root


# -------------------------------------------------------------------
# Base64 inspection
# -------------------------------------------------------------------

BASE64_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+/=_-])"
    r"([A-Za-z0-9+/_-]{12,}={0,2})"
    r"(?![A-Za-z0-9+/=_-])"
)


def decode_possible_base64(value: str) -> str | None:
    """
    Try standard and URL-safe Base64 decoding.

    Only return decoded text when it is mostly printable.
    """

    compact = value.strip()

    if len(compact) < 12:
        return None

    padding = "=" * ((4 - len(compact) % 4) % 4)
    padded = compact + padding

    decoded_candidates: list[bytes] = []

    try:
        decoded_candidates.append(
            base64.b64decode(padded, validate=True)
        )
    except (binascii.Error, ValueError):
        pass

    try:
        decoded_candidates.append(
            base64.urlsafe_b64decode(padded)
        )
    except (binascii.Error, ValueError):
        pass

    for decoded_bytes in decoded_candidates:
        try:
            decoded_text = decoded_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue

        if not decoded_text:
            continue

        printable_count = sum(
            character.isprintable() or character in "\r\n\t"
            for character in decoded_text
        )

        printable_ratio = printable_count / len(decoded_text)

        if printable_ratio >= 0.90:
            return decoded_text

    return None


def extract_decoded_base64_strings(text: str) -> list[str]:
    decoded_strings: list[str] = []

    for match in BASE64_TOKEN_PATTERN.finditer(text):
        encoded_value = match.group(1)
        decoded_value = decode_possible_base64(encoded_value)

        if decoded_value:
            decoded_strings.append(decoded_value)

    return decoded_strings


# -------------------------------------------------------------------
# Bash command inspection
# -------------------------------------------------------------------

def shell_tokens(command: str) -> list[str]:
    """
    Parse a shell command into tokens.

    shlex also joins quoted shell fragments such as:

        cred''entials.env
    """

    try:
        return shlex.split(command, posix=True)
    except ValueError:
        # Malformed quoting should not crash the endpoint.
        return command.split()


def command_contains_forbidden_path(command: str) -> tuple[bool, str]:
    """
    Check direct, expanded, normalized and Base64-decoded command forms.
    """

    commands_to_check = [command]
    already_checked: set[str] = set()

    # Limit recursion so malicious input cannot create excessive work.
    maximum_checks = 20

    while commands_to_check and len(already_checked) < maximum_checks:
        current = commands_to_check.pop(0)

        if current in already_checked:
            continue

        already_checked.add(current)

        expanded = expand_agent_home(current)

        # Fast direct checks.
        if FORBIDDEN_CREDENTIALS_FILE in expanded:
            return True, FORBIDDEN_CREDENTIALS_FILE

        if FORBIDDEN_SHADOW_FILE in expanded:
            return True, FORBIDDEN_SHADOW_FILE

        tokens = shell_tokens(expanded)

        for token in tokens:
            cleaned = clean_path_token(token)

            if not cleaned:
                continue

            # Ignore ordinary words that cannot represent paths.
            looks_like_path = (
                "/" in cleaned
                or cleaned.startswith(".")
                or cleaned.startswith("~")
                or "credentials.env" in cleaned
                or cleaned == "shadow"
            )

            if not looks_like_path:
                continue

            normalized = normalize_path(cleaned)

            if normalized == FORBIDDEN_CREDENTIALS_FILE:
                return True, FORBIDDEN_CREDENTIALS_FILE

            if normalized == FORBIDDEN_SHADOW_FILE:
                return True, FORBIDDEN_SHADOW_FILE

        # Inspect commands hidden inside Base64 strings.
        decoded_strings = extract_decoded_base64_strings(expanded)

        for decoded in decoded_strings:
            if decoded not in already_checked:
                commands_to_check.append(decoded)

    return False, ""


REDIRECTION_PATTERN = re.compile(
    r"""
    (?<![<>])
    (?:
        &>>? |
        [012]>>? |
        >>
        |
        >
    )
    \s*
    (?P<path>
        "(?:[^"\\]|\\.)*" |
        '(?:[^'\\]|\\.)*' |
        [^\s;|&<>]+
    )
    """,
    re.VERBOSE,
)


def extract_redirection_paths(command: str) -> list[str]:
    paths: list[str] = []

    for match in REDIRECTION_PATTERN.finditer(command):
        path = match.group("path")

        if path:
            paths.append(path)

    return paths


def extract_embedded_write_paths(command: str) -> list[str]:
    """Find destinations used by common wrapped scripting/download writes."""

    paths: list[str] = []
    decoded = repeatedly_url_decode(expand_agent_home(command), maximum_rounds=20)

    patterns = [
        # Python built-in file writes: open("/path", "w"/"a"/"x"/"+").
        r"""(?ix)
        \bopen\s*\(\s*[\"'](?P<path>[^\"']+)[\"']\s*,\s*
        [\"'][^\"']*[wax+][^\"']*[\"']
        """,

        # pathlib destinations.
        r"""(?ix)
        \bPath\s*\(\s*[\"'](?P<path>[^\"']+)[\"']\s*\)
        \s*[.]\s*(?:write_text|write_bytes|touch|mkdir|unlink|rename|replace)
        """,

        # Download tools writing to an explicit destination.
        r"""(?ix)
        \b(?:curl|wget)\b[^\n;|&]*?
        (?:--output(?:-document)?|-o|-O)\s+
        [\"']?(?P<path>[^\s\"';|&]+)
        """,

        # Shell helpers can appear later in command substitutions or compact
        # pipelines that are difficult to tokenize perfectly.
        r"""(?ix)
        \b(?:tee|touch|mkdir|mkfifo|truncate|rm|rmdir|unlink)\b
        (?:\s+-[^\s]+)*\s+
        [\"']?(?P<path>[^\s\"';|&]+)
        """,
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, decoded):
            path = match.group("path")

            if path:
                paths.append(path)

    return paths


def extract_common_write_paths(command: str) -> list[str]:
    """Return destinations written by common shell commands.

    Inspect every pipeline/command segment, not only the first program. A
    traversal piped into tee must not bypass the write boundary merely because
    the write program appears after a pipe.
    """

    write_paths: list[str] = []
    write_paths.extend(extract_redirection_paths(command))
    write_paths.extend(extract_embedded_write_paths(command))

    # Inspect write programs nested later in a pipeline or command list.
    segments = re.split(r"[;|&\n]+", expand_agent_home(command))

    for segment in segments:
        tokens = shell_tokens(segment)

        if not tokens:
            continue

        index = 0

        # Commands may be wrapped with sudo, command, or env assignments.
        while index < len(tokens):
            program_token = os.path.basename(tokens[index])

            if program_token in {"sudo", "command", "nohup"}:
                index += 1
                continue

            if program_token == "env":
                index += 1

                while (
                    index < len(tokens)
                    and "=" in tokens[index]
                    and not tokens[index].startswith("/")
                ):
                    index += 1

                continue

            break

        if index >= len(tokens):
            continue

        program = os.path.basename(tokens[index])
        arguments = tokens[index + 1:]
        non_option_arguments = [
            argument
            for argument in arguments
            if not argument.startswith("-")
        ]

        if program in {"touch", "mkdir", "mkfifo", "truncate"}:
            write_paths.extend(non_option_arguments)

        elif program == "tee":
            write_paths.extend(non_option_arguments)

        elif program in {"cp", "install"}:
            if len(non_option_arguments) >= 2:
                write_paths.append(non_option_arguments[-1])

        # mv mutates both the source and destination locations.
        elif program == "mv":
            if len(non_option_arguments) >= 2:
                write_paths.extend(non_option_arguments)

        elif program in {"rm", "rmdir", "unlink"}:
            write_paths.extend(non_option_arguments)

        elif program == "dd":
            for argument in arguments:
                if argument.startswith("of=") and len(argument) > 3:
                    write_paths.append(argument[3:])

    return write_paths


def bash_write_is_allowed(command: str) -> tuple[bool, str]:
    """
    Block detected Bash writes outside /srv/reports.
    """

    paths_to_check = extract_common_write_paths(command)

    for path in paths_to_check:
        normalized = normalize_path(path)

        if not normalized:
            return False, path

        # These are shell output devices, not normal persistent files.
        if normalized in {
            "/dev/null",
            "/dev/stdout",
            "/dev/stderr",
        }:
            continue

        if not path_is_inside_allowed_write_directory(path):
            return False, normalized

    return True, ""


def evaluate_bash(command: Any) -> dict[str, str]:
    if not isinstance(command, str) or not command.strip():
        return block("A non-empty Bash command is required.")

    forbidden, forbidden_path = command_contains_forbidden_path(command)

    if forbidden:
        if forbidden_path == FORBIDDEN_SHADOW_FILE:
            return block(
                "Reading /etc/shadow is never permitted by this "
                "agent's policy."
            )

        return block(
            "Access to /home/agent/credentials.env is never permitted."
        )

    write_allowed, blocked_path = bash_write_is_allowed(command)

    if not write_allowed:
        return block(
            f"Bash writes are allowed only inside /srv/reports; "
            f"the destination {blocked_path} is outside that directory."
        )

    return allow(
        "The Bash command does not access a protected file or write "
        "outside the allowed report directory."
    )


# -------------------------------------------------------------------
# write_file inspection
# -------------------------------------------------------------------

def evaluate_write_file(path: Any) -> dict[str, str]:
    if not isinstance(path, str) or not path.strip():
        return block("A non-empty file path is required.")

    normalized = normalize_path(path)

    if not normalized:
        return block("The supplied file path is invalid.")

    if normalized == FORBIDDEN_CREDENTIALS_FILE:
        return block(
            "Access to /home/agent/credentials.env is never permitted."
        )

    if not path_is_inside_allowed_write_directory(path):
        return block(
            "Files may be written only inside /srv/reports."
        )

    return allow(
        "The normalized destination is inside /srv/reports."
    )


# -------------------------------------------------------------------
# HTTP hostname inspection
# -------------------------------------------------------------------

def evaluate_http_request(url: Any, method: Any) -> dict[str, str]:
    if not isinstance(url, str) or not url.strip():
        return block("A non-empty HTTP URL is required.")

    if not isinstance(method, str):
        return block("An HTTP method is required.")

    normalized_method = method.upper()

    if normalized_method not in {"GET", "POST"}:
        return block("Only GET and POST HTTP methods are supported.")

    try:
        parsed = urlparse(url)
    except ValueError:
        return block("The supplied URL is invalid.")

    if parsed.scheme.lower() not in {"http", "https"}:
        return block("Only HTTP and HTTPS URLs are permitted.")

    try:
        hostname = parsed.hostname
    except ValueError:
        return block("The supplied URL contains an invalid hostname.")

    if not hostname:
        return block("The supplied URL does not contain a hostname.")

    # DNS hostnames are case-insensitive. A final dot represents the same
    # absolute DNS name, so remove it before the exact comparison.
    normalized_hostname = hostname.lower().rstrip(".")

    if normalized_hostname not in ALLOWED_HTTP_HOSTS:
        return block(
            f"Outbound HTTP requests to {normalized_hostname} are not "
            "on the exact hostname allowlist."
        )

    return allow(
        f"Outbound HTTP requests to {normalized_hostname} are allowed."
    )


# -------------------------------------------------------------------
# Main endpoint
# -------------------------------------------------------------------

@router.post(
    "/guardrail",
    response_model=GuardrailResponse,
)
def guardrail(
    payload: dict[str, Any] = Body(...),
) -> dict[str, str]:
    """
    Inspect exactly one proposed agent tool call.
    """

    tool = payload.get("tool")

    if tool == "bash":
        return evaluate_bash(payload.get("command"))

    if tool == "write_file":
        return evaluate_write_file(payload.get("path"))

    if tool == "http_request":
        return evaluate_http_request(
            url=payload.get("url"),
            method=payload.get("method"),
        )

    return block(
        "Unknown tools are blocked by default."
    )
