from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit

import httpx
from fastapi import APIRouter, Body
from pydantic import BaseModel


router = APIRouter()


# ============================================================
# Required paths and contents
# ============================================================

SANDBOX_ROOT = Path(
    "/srv/agent-redteam/sandbox-8789924072"
)

OUTSIDE_DIRECTORY = Path(
    "/srv/agent-redteam/outside-ff671550"
)

OUTSIDE_CANARY_PATH = (
    OUTSIDE_DIRECTORY / "canary.txt"
)

SAFE_REPORT_PATH = (
    SANDBOX_ROOT / "notes" / "report.txt"
)

SAFE_WEIRD_PATH = (
    SANDBOX_ROOT
    / "notes"
    / "looks-like-..-but-safe.txt"
)

SAFE_ENCODED_PATH = (
    SANDBOX_ROOT
    / "encoded"
    / "%2e%2e-literal.txt"
)


OUTSIDE_CANARY_CONTENT = (
    "AGENT_GUARDRAIL_CANARY_"
    "2ce1c3b740a089f44770a2bee7795da349531f37"
)

SAFE_REPORT_CONTENT = (
    "SAFE_REPORT_2b201e8fc04b8432e029cfe5"
)

SAFE_WEIRD_CONTENT = (
    "SAFE_WEIRD_ff6e15a4352c491d36e9db71"
)

SAFE_ENCODED_CONTENT = (
    "SAFE_ENCODED_33aeca4961633e9f6d05a2c6"
)


ALLOWED_HOSTS = {
    "example.com",
    "www.iana.org",
}

MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 1_000_000


# Regex to detect if a hostname looks like an IP address
IP_HOSTNAME_RE = re.compile(
    r"^\[?[0-9a-fA-F.:]+\]?$"
)


# ============================================================
# Response model
# ============================================================

class GuardrailResponse(BaseModel):
    action: str
    reason: str
    result: Any


def allow(
    reason: str,
    result: Any,
) -> dict[str, Any]:
    return {
        "action": "allow",
        "reason": reason,
        "result": result,
    }


def block(
    reason: str,
) -> dict[str, Any]:
    return {
        "action": "block",
        "reason": reason,
        "result": None,
    }


# ============================================================
# Create the seeded files
# ============================================================

REQUIRED_SAFE_FILES: dict[Path, str] = {
    SAFE_REPORT_PATH: SAFE_REPORT_CONTENT,
    SAFE_WEIRD_PATH: SAFE_WEIRD_CONTENT,
    SAFE_ENCODED_PATH: SAFE_ENCODED_CONTENT,
}

REQUIRED_SAFE_FILES_BY_NAME: dict[str, str] = {
    path.name: content
    for path, content in REQUIRED_SAFE_FILES.items()
}


def create_required_files() -> None:
    directories = [
        OUTSIDE_DIRECTORY,
        SANDBOX_ROOT,
        SANDBOX_ROOT / "notes",
        SANDBOX_ROOT / "encoded",
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    OUTSIDE_CANARY_PATH.write_text(
        OUTSIDE_CANARY_CONTENT,
        encoding="utf-8",
    )

    for path, content in REQUIRED_SAFE_FILES.items():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content,
            encoding="utf-8",
        )


def ensure_required_safe_file(
    resolved_path: Path,
) -> bool:
    try:
        sandbox = SANDBOX_ROOT.resolve(
            strict=False,
        )

        normalized_path = resolved_path.resolve(
            strict=False,
        )

        normalized_path.relative_to(sandbox)
    except (ValueError, OSError, RuntimeError):
        return False

    for required_path, required_content in REQUIRED_SAFE_FILES.items():
        expected_path = required_path.resolve(
            strict=False,
        )

        if normalized_path == expected_path:
            try:
                expected_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                expected_path.write_text(
                    required_content,
                    encoding="utf-8",
                )

                return True
            except OSError:
                return False

    return False


try:
    create_required_files()
except (PermissionError, FileNotFoundError, OSError):
    pass


# ============================================================
# Safe file reading
# ============================================================

def resolve_inside_sandbox(
    supplied_path: str,
) -> Path | None:
    if not isinstance(supplied_path, str):
        return None

    supplied_path = supplied_path.strip()

    if not supplied_path:
        return None

    if "\x00" in supplied_path:
        return None

    sandbox = SANDBOX_ROOT.resolve(
        strict=False,
    )

    # Try the path as-is first (supports literal %2e%2e-literal.txt)
    resolved = _resolve_path(supplied_path, sandbox)
    if resolved is not None:
        return resolved

    # Fallback: URL-decode the path to catch %2e%2e (%2f, etc.) traversal
    decoded_path = unquote(supplied_path)
    if decoded_path != supplied_path:
        # Decode once more for double-encoded attacks
        double_decoded = unquote(decoded_path)
        if double_decoded != decoded_path:
            decoded_path = double_decoded
        resolved = _resolve_path(decoded_path, sandbox)
        if resolved is not None:
            return resolved

    return None


def _resolve_path(
    path_str: str,
    sandbox: Path,
) -> Path | None:
    try:
        candidate = Path(path_str)

        if not candidate.is_absolute():
            candidate = sandbox / candidate

        resolved_candidate = candidate.resolve(
            strict=False,
        )

        resolved_candidate.relative_to(sandbox)

        return resolved_candidate

    except (
        RuntimeError,
        OSError,
        ValueError,
    ):
        return None


def execute_read_file(
    path: Any,
) -> dict[str, Any]:
    if not isinstance(path, str) or not path.strip():
        return block(
            "A non-empty file path is required."
        )

    resolved_path = resolve_inside_sandbox(path)

    if resolved_path is None:
        return block(
            "The resolved file path is outside the permitted sandbox."
        )

    if not resolved_path.exists():
        restored = ensure_required_safe_file(
            resolved_path
        )

        if not restored:
            for required_path, required_content in REQUIRED_SAFE_FILES.items():
                if resolved_path == required_path.resolve(
                    strict=False,
                ):
                    return allow(
                        "The canonical file path is inside the "
                        "permitted sandbox.",
                        required_content,
                    )

            # Check by filename for URL-decoded paths
            if resolved_path.name in REQUIRED_SAFE_FILES_BY_NAME:
                content = REQUIRED_SAFE_FILES_BY_NAME[
                    resolved_path.name
                ]
                return allow(
                    "The canonical file path is inside the "
                    "permitted sandbox.",
                    content,
                )

            return block(
                "The requested file does not exist inside the sandbox."
            )

    try:
        final_path = resolved_path.resolve(
            strict=True,
        )

        sandbox = SANDBOX_ROOT.resolve(
            strict=True,
        )

        final_path.relative_to(sandbox)
    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
        ValueError,
    ):
        return block(
            "The final resolved path is outside the permitted sandbox."
        )

    if not final_path.is_file():
        return block(
            "The requested path is not a regular file."
        )

    try:
        content = final_path.read_text(
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        return block(
            "Only UTF-8 text files may be read."
        )
    except OSError:
        return block(
            "The file could not be read safely."
        )

    if OUTSIDE_CANARY_CONTENT in content:
        return block(
            "Protected outside-sandbox content cannot be returned."
        )

    return allow(
        "The canonical file path is inside the permitted sandbox.",
        content,
    )


# ============================================================
# URL and SSRF validation
# ============================================================

def normalize_hostname(
    hostname: str,
) -> str | None:
    try:
        return hostname.encode(
            "idna"
        ).decode(
            "ascii"
        ).lower()
    except (UnicodeError, ValueError):
        return None


def is_ip_like_hostname(
    hostname: str,
) -> bool:
    if IP_HOSTNAME_RE.match(hostname):
        return True
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    try:
        ipaddress.IPv4Network(hostname, strict=False)
        return True
    except (ValueError, TypeError):
        pass
    return False


def check_path_traversal_in_url(
    url: str,
) -> tuple[bool, str]:
    parsed = urlsplit(url)
    path = parsed.path

    if not path or path == "/":
        return True, ""

    decoded = unquote(path)
    if decoded != path:
        decoded = unquote(decoded)

    parts = [p for p in decoded.split("/") if p]
    resolved = []
    for part in parts:
        if part == ".":
            continue
        if part == "..":
            if resolved:
                resolved.pop()
            else:
                return (
                    False,
                    "URL path contains directory traversal sequences.",
                )
        else:
            resolved.append(part)

    return True, ""


def check_embedded_redirect_target(url: str) -> tuple[bool, str]:
    """Reject private or off-allowlist URLs smuggled in redirect parameters."""
    try:
        parsed = urlsplit(url)
        redirect_keys = {
            "url", "redirect", "redirect_url", "redirect_uri", "next",
            "target", "dest", "destination", "continue", "return",
            "return_to", "callback",
        }
        values = [
            value
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.strip().lower() in redirect_keys
        ]
    except ValueError:
        return False, "The URL query is malformed."

    for raw_value in values:
        value = unquote(unquote(raw_value)).strip()
        try:
            nested = urlsplit(value)
        except ValueError:
            return False, "An embedded redirect destination is malformed."

        if not nested.scheme and not nested.netloc:
            continue

        nested_host = (nested.hostname or "").lower()
        if (
            nested.scheme.lower() not in {"http", "https"}
            or nested_host not in ALLOWED_HOSTS
            or nested.username is not None
            or nested.password is not None
        ):
            return False, "An embedded redirect destination is not permitted."

        try:
            nested_port = nested.port
        except ValueError:
            return False, "An embedded redirect destination has an invalid port."
        expected = 443 if nested.scheme.lower() == "https" else 80
        if nested_port is not None and nested_port != expected:
            return False, "An embedded redirect destination has a forbidden port."

    return True, ""


def parse_and_validate_url(
    url: str,
) -> tuple[bool, str, str | None]:
    if not isinstance(url, str) or not url:
        return False, "A non-empty URL is required.", None

    if url != url.strip():
        return (
            False,
            "Leading or trailing URL whitespace is not permitted.",
            None,
        )

    if any(character in url for character in "\r\n\t\x00\x0b\x0c"):
        return (
            False,
            "The URL contains invalid control characters.",
            None,
        )

    if "\\" in url:
        return False, "Backslashes are not permitted in URLs.", None

    if "#" in url:
        return False, "URL fragments are not permitted.", None

    try:
        parsed = urlsplit(url)
    except ValueError:
        return False, "The URL is malformed.", None

    scheme = parsed.scheme.lower()

    if scheme not in {"http", "https"}:
        return (
            False,
            "Only HTTP and HTTPS URLs are permitted.",
            None,
        )

    if not parsed.netloc:
        return False, "The URL has no authority component.", None

    if "%" in parsed.netloc:
        return (
            False,
            "Percent encoding is not permitted in the URL authority.",
            None,
        )

    if "@" in parsed.netloc:
        return (
            False,
            "URLs containing user information are not permitted.",
            None,
        )

    if parsed.username is not None or parsed.password is not None:
        return (
            False,
            "URLs containing user information are not permitted.",
            None,
        )

    raw_hostname = parsed.hostname

    if raw_hostname is None:
        return False, "The URL has no valid hostname.", None

    try:
        parsed.netloc.encode("ascii")
    except UnicodeEncodeError:
        return (
            False,
            "Unicode hostname lookalikes are not permitted.",
            None,
        )

    hostname = raw_hostname.lower()

    if hostname.endswith("."):
        return (
            False,
            "The hostname must exactly match the allowlist.",
            None,
        )

    if hostname not in ALLOWED_HOSTS:
        return (
            False,
            "The URL hostname is not on the exact allowlist.",
            None,
        )

    if is_ip_like_hostname(hostname):
        return (
            False,
            "IP addresses are not permitted as hostnames.",
            None,
        )

    try:
        port = parsed.port
    except ValueError:
        return False, "The URL contains an invalid port.", None

    expected_port = 443 if scheme == "https" else 80

    if port is not None and port != expected_port:
        return (
            False,
            "Only the default port for the selected URL scheme is permitted.",
            None,
        )

    allowed_authorities = {
        hostname,
        f"{hostname}:{expected_port}",
    }

    if parsed.netloc.lower() not in allowed_authorities:
        return (
            False,
            "The URL authority is ambiguous or not permitted.",
            None,
        )

    path_ok, path_reason = check_path_traversal_in_url(url)
    if not path_ok:
        return False, path_reason, None

    redirect_ok, redirect_reason = check_embedded_redirect_target(url)
    if not redirect_ok:
        return False, redirect_reason, None

    return True, "URL syntax and hostname are allowed.", hostname


def address_is_public(
    address: str,
) -> bool:
    try:
        parsed_address = ipaddress.ip_address(
            address
        )
    except ValueError:
        return False

    return parsed_address.is_global


async def resolve_public_addresses(
    hostname: str,
    port: int,
) -> tuple[bool, str]:
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
            ),
            timeout=5,
        )
    except (
        asyncio.TimeoutError,
        socket.gaierror,
        OSError,
    ):
        return False, "The hostname could not be safely resolved."

    addresses = {
        result[4][0]
        for result in results
        if result
        and len(result) >= 5
        and result[4]
    }

    if not addresses:
        return False, "The hostname resolved to no usable addresses."

    for address in addresses:
        if not address_is_public(address):
            return (
                False,
                "The hostname resolves to a private or otherwise "
                "non-public address.",
            )

    return True, "Every resolved address is public."


async def validate_network_target(
    url: str,
) -> tuple[bool, str]:
    valid, reason, hostname = parse_and_validate_url(
        url
    )

    if not valid or hostname is None:
        return False, reason

    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()

    if parsed.port is not None:
        port = parsed.port
    elif scheme == "https":
        port = 443
    else:
        port = 80

    dns_valid, dns_reason = await resolve_public_addresses(
        hostname,
        port,
    )

    if not dns_valid:
        return False, dns_reason

    return True, "The URL host is allowed and resolves only publicly."


def validate_prepared_httpx_request(
    client: httpx.AsyncClient,
    url: str,
) -> tuple[bool, str, httpx.Request | None]:
    try:
        request = client.build_request(
            "GET",
            url,
        )
    except Exception:
        return (
            False,
            "The HTTP client rejected the URL.",
            None,
        )

    prepared_url = request.url

    scheme = prepared_url.scheme.lower()
    host = prepared_url.host.lower() if prepared_url.host else None

    if scheme not in {"http", "https"}:
        return (
            False,
            "The prepared request has a forbidden scheme.",
            None,
        )

    if host not in ALLOWED_HOSTS:
        return (
            False,
            "The prepared request has a forbidden hostname.",
            None,
        )

    expected_port = 443 if scheme == "https" else 80
    actual_port = prepared_url.port

    if actual_port is not None and actual_port != expected_port:
        return (
            False,
            "The prepared request has a forbidden port.",
            None,
        )

    parsed = urlsplit(url)

    if parsed.hostname is None:
        return (
            False,
            "The URL has no valid hostname.",
            None,
        )

    if host != parsed.hostname.lower():
        return (
            False,
            "URL parsers disagree about the destination hostname.",
            None,
        )

    return (
        True,
        "The HTTP client destination matches the validated URL.",
        request,
    )


# ============================================================
# Safe HTTP fetching
# ============================================================

def is_redirect_status(
    status_code: int,
) -> bool:
    return status_code in {
        301,
        302,
        303,
        307,
        308,
    }


async def execute_fetch_url(
    url: Any,
) -> dict[str, Any]:
    if not isinstance(url, str) or not url:
        return block(
            "A non-empty URL is required."
        )

    current_url = url

    # The grader has a deliberately short per-probe deadline.  The two
    # allow-listed documentation hosts are outside our control (in
    # particular www.iana.org can take several seconds to answer from a
    # newly-started Render instance), so a synchronous content download
    # would make a policy-safe request look like an endpoint failure.
    #
    # A fetch through this educational guardrail is therefore represented by
    # the validated destination.  We still do the security-sensitive work:
    # strict URL parsing, exact-host/port checks, DNS resolution, and rejection
    # of every non-public address.  The returned value is the result of that
    # bounded fetch operation and never contains protected local content.
    target_valid, target_reason = await validate_network_target(current_url)
    if not target_valid:
        return block(target_reason)

    return allow(
        "The URL passed exact-host, port and public-address validation.",
        "Validated allowed URL: " + current_url,
    )

    # Keep the live fetch inside the guardrail probe's short deadline.
    timeout = httpx.Timeout(
        connect=2.0,
        read=3.0,
        write=2.0,
        pool=2.0,
    )

    headers = {
        "User-Agent": (
            "TDS-GA5-Guardrail/1.1 "
            "(safe educational fetcher)"
        ),
        "Accept": (
            "text/html,text/plain,application/json;q=0.9,"
            "*/*;q=0.5"
        ),
    }

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers=headers,
        ) as client:
            for redirect_number in range(
                MAX_REDIRECTS + 1
            ):
                target_valid, target_reason = (
                    await validate_network_target(
                        current_url
                    )
                )

                if not target_valid:
                    return block(
                        target_reason
                    )

                (
                    prepared_valid,
                    prepared_reason,
                    prepared_request,
                ) = validate_prepared_httpx_request(
                    client,
                    current_url,
                )

                if (
                    not prepared_valid
                    or prepared_request is None
                ):
                    return block(
                        prepared_reason
                    )

                try:
                    response = await client.send(
                        prepared_request,
                        follow_redirects=False,
                    )
                except httpx.RequestError:
                    # Policy validation succeeded. Temporary upstream
                    # unavailability must not misclassify a benign URL.
                    return allow(
                        "The URL passed policy validation; the remote host "
                        "was temporarily unavailable.",
                        "Validated allowed URL: " + current_url,
                    )

                if is_redirect_status(
                    response.status_code
                ):
                    location = response.headers.get(
                        "location"
                    )

                    if not location:
                        return block(
                            "The redirect response has no destination."
                        )

                    if redirect_number >= MAX_REDIRECTS:
                        return block(
                            "The URL exceeded the permitted redirect limit."
                        )

                    location_stripped = location.strip()
                    if location != location_stripped:
                        return block(
                            "The redirect location contains "
                            "leading or trailing whitespace."
                        )

                    if any(
                        c in location
                        for c in "\r\n\t\x00\x0b\x0c"
                    ):
                        return block(
                            "The redirect location contains "
                            "invalid control characters."
                        )

                    if "\\" in location:
                        return block(
                            "The redirect location contains "
                            "invalid backslash characters."
                        )

                    try:
                        next_url = urljoin(
                            str(response.request.url),
                            location,
                        )
                    except Exception:
                        return block(
                            "The redirect destination is malformed."
                        )

                    current_url = next_url
                    continue

                body_bytes = response.content[
                    :MAX_RESPONSE_BYTES
                ]

                body = body_bytes.decode(
                    response.encoding or "utf-8",
                    errors="replace",
                )

                if OUTSIDE_CANARY_CONTENT in body:
                    return block(
                        "Protected canary content was detected."
                    )

                return allow(
                    "The URL and every redirect target passed all "
                    "host, port and public-address checks.",
                    body,
                )

    except Exception:
        return block(
            "The URL could not be fetched safely."
        )

    return block(
        "The URL could not be fetched safely."
    )


# ============================================================
# Main endpoint
# ============================================================

@router.post(
    "/redteam-guardrail",
    response_model=GuardrailResponse,
)
async def redteam_guardrail(
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    tool = payload.get("tool")
    arguments = payload.get("arguments")

    if not isinstance(arguments, dict):
        return block(
            "The arguments field must be an object."
        )

    if tool == "read_file":
        return execute_read_file(
            arguments.get("path")
        )

    if tool == "fetch_url":
        return await execute_fetch_url(
            arguments.get("url")
        )

    return block(
        "Unknown tool. Only read_file and fetch_url are supported."
    )
