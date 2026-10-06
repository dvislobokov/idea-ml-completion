"""Secret / personal-data detection and scrubbing for the code-completion training corpus.

Single source of truth for the patterns: `scan_secrets.py` (corpus audit) and the encode-time filter both use
`find_matches()`.  Pure stdlib, no I/O.  Works on `str`; `scrub_bytes()` is the convenience wrapper for the
`bytes` that `encode_corpus.py` reads.

Public API
    find_matches(text, path="") -> list[Match]         every detection (including report-only categories)
    classify_line(line)         -> None | category     highest-priority category found in one line
    scrub(text, path="", policy=DEFAULT_POLICY) -> (text | None, stats)
    scrub_bytes(data, path="", policy=DEFAULT_POLICY) -> (bytes | None, stats)
    mask(match)                 -> redacted sample for reports (never the full secret)
    is_test_path(path)          -> True for test / fixture / example files

Categories (priority order; higher first when spans overlap)
    private_key   -----BEGIN ... PRIVATE KEY-----  (file is dropped by the default policy)
    cloud_key     vendor-prefixed tokens: AWS, Google, GitHub, GitLab, Slack, Stripe, Twilio, SendGrid, npm, PyPI,
                  OpenAI, Telegram, Discord, Azure storage, Shopify, Mailgun, DigitalOcean, HashiCorp, age, ...
    jwt           eyJ....eyJ....sig
    conn_string   credentials inside URLs (scheme://user:pass@host), Go mysql DSNs (user:pass@tcp(...)),
                  C#/ADO connection strings (Server=...;Password=...)
    credential    generic assignments: password|secret|token|api_key ... = "value" / key=value in URLs /
                  "Authorization: Basic|Bearer xxx"
    high_entropy  base64-like literal, len >= 32, Shannon entropy > 4.5 bits/char, mixed case + digits,
                  not a known data blob (PNG/gzip/PDF/zip/DER/PEM body/protobuf descriptor)
    email         e-mail address not on the allowlist (placeholders, example.*, localhost, noreply, git@, licence
                  and author headers)
    public_ip     IPv4 outside private/loopback/link-local/documentation/multicast/reserved ranges, not a
                  well-known public resolver and not a version-like x.y.0.0
    hash_like     REPORT ONLY: 32/40/64/96/128 hex digits (md5/sha1/sha256/sha384/sha512 test vectors, git shas)
    email_allow   REPORT ONLY: allowlisted e-mail
    ip_wellknown  REPORT ONLY: 1.1.1.1, 8.8.8.8, 1.2.3.4 and friends
    phone         REPORT ONLY: phone-like number
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = ["Match", "find_matches", "classify_line", "scrub", "scrub_matches", "scrub_bytes", "mask", "is_test_path",
           "DEFAULT_POLICY", "SCRUBBED", "REPORT_ONLY", "CREDENTIAL_CATEGORIES", "CATEGORIES"]

# --------------------------------------------------------------------------------------------------------------
# categories

CATEGORIES = ["private_key", "cloud_key", "jwt", "conn_string", "credential", "high_entropy", "email",
              "public_ip", "hash_like", "email_allow", "ip_wellknown", "phone"]
PRIORITY = {c: i for i, c in enumerate(CATEGORIES)}
SCRUBBED = {"private_key", "cloud_key", "jwt", "conn_string", "credential", "high_entropy", "email", "public_ip"}
REPORT_ONLY = {"hash_like", "email_allow", "ip_wellknown", "phone"}
# matches that count towards the ">= N credential hits -> drop file" rule
CREDENTIAL_CATEGORIES = {"cloud_key", "jwt", "conn_string", "credential"}

PLACEHOLDER = {
    "private_key": "<redacted-private-key>",
    "cloud_key": "<redacted-key>",
    "jwt": "<redacted-jwt>",
    "conn_string": "<redacted-password>",
    "credential": "<redacted-secret>",
    "high_entropy": "<redacted-blob>",
    "email": "user@example.com",
    "public_ip": "203.0.113.1",
}


@dataclass
class Match:
    category: str
    subtype: str
    start: int          # span of the VALUE that gets replaced (not the key / scheme / host)
    end: int
    line: int = 0       # 1-based, filled by find_matches
    extra: dict = field(default_factory=dict)

    def value(self, text: str) -> str:
        return text[self.start:self.end]


# --------------------------------------------------------------------------------------------------------------
# path classification

_TEST_PATH = re.compile(
    r"(?i)(?:^|/)(?:test|tests|testing|testdata|test_data|fixtures?|examples?|samples?|mocks?|_examples?|e2e|"
    r"integration[-_]?tests?|unittests?|benchmarks?|spec|specs|__tests__|[\w.-]*(?:[._-]|unit|integration|functional|acceptance)tests?)(?:/|$)"
    r"|_test\.go$|_mock\.go$|example_[\w]*\.go$"
    r"|(?:tests?|spec|specs|fixture|fixtures|mock|mocks|fake|fakes|stub|stubs|example|examples|sample|samples)\.cs$"
    r"|(?:^|/)(?:test|mock|fake|stub)[\w]*\.cs$")


def is_test_path(path: str) -> bool:
    return bool(_TEST_PATH.search(path.replace("\\", "/")))


# --------------------------------------------------------------------------------------------------------------
# patterns

_B64 = r"[A-Za-z0-9+/=_-]"

PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9 ]{0,30} )?PRIVATE KEY(?: BLOCK)?-----")
# allowed between BEGIN and END: base64 lines, PEM headers (Proc-Type:, DEK-Info:), Go/C# string plumbing (\n " + `)
_PEM_BODY = re.compile(r"[A-Za-z0-9+/=\s\\\"'`+,:;()-]*")

# (subtype, regex, needles): regex runs only when one of the needles occurs in the text (lower-cased text for
# case-insensitive needles, marked by a leading "~"); the whole match is the value unless the regex has a group "v"
CLOUD_KEYS = [
    ("aws_access_key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[A-Z0-9]{16}(?![A-Z0-9])"), ("AKIA", "ASIA", "AGPA", "AIDA", "AROA", "ANPA", "ANVA")),
    ("aws_secret_key", re.compile(r"(?i)aws[\w.\-]{0,30}?secret[\w.\-]{0,20}?\s*[:=]\s*[\"'`]?(?P<v>[A-Za-z0-9/+]{40})(?![A-Za-z0-9/+])"), ("~aws",)),
    ("google_api_key", re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"), ("AIza",)),
    ("google_oauth_secret", re.compile(r"(?<![A-Za-z0-9_-])GOCSPX-[0-9A-Za-z_-]{28}(?![A-Za-z0-9_-])"), ("GOCSPX-",)),
    ("google_oauth_token", re.compile(r"(?<![A-Za-z0-9_.-])ya29\.[0-9A-Za-z_-]{30,}(?![A-Za-z0-9_-])"), ("ya29.",)),
    ("github_token", re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36,255}(?![A-Za-z0-9])"), ("ghp_", "gho_", "ghu_", "ghs_", "ghr_")),
    ("github_pat", re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{22,255}(?![A-Za-z0-9_])"), ("github_pat_",)),
    ("gitlab_pat", re.compile(r"(?<![A-Za-z0-9_-])glpat-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"), ("glpat-",)),
    ("slack_token", re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-[0-9]{8,}-[0-9A-Za-z-]{8,}(?![A-Za-z0-9_-])"), ("xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-")),
    ("slack_webhook", re.compile(r"hooks\.slack\.com/services/(?P<v>T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{20,})"), ("hooks.slack.com",)),
    ("stripe_key", re.compile(r"(?<![A-Za-z0-9_])[sr]k_live_[0-9a-zA-Z]{20,}(?![A-Za-z0-9])"), ("sk_live_", "rk_live_")),
    ("twilio_sid", re.compile(r"(?<![A-Za-z0-9])(?:AC|SK)[0-9a-f]{32}(?![A-Za-z0-9])"), ("AC", "SK")),
    ("sendgrid_key", re.compile(r"(?<![A-Za-z0-9_-])SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])"), ("SG.",)),
    ("npm_token", re.compile(r"(?<![A-Za-z0-9_])npm_[A-Za-z0-9]{36}(?![A-Za-z0-9])"), ("npm_",)),
    ("pypi_token", re.compile(r"(?<![A-Za-z0-9_-])pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{50,}"), ("pypi-AgEIcHlwaS5vcmc",)),
    ("openai_key", re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9_-]{20,}T3BlbkFJ[A-Za-z0-9_-]{20,}"), ("T3BlbkFJ",)),
    ("telegram_bot", re.compile(r"(?<![0-9A-Za-z_-])[0-9]{8,10}:AA[A-Za-z0-9_-]{33}(?![A-Za-z0-9_-])"), (":AA",)),
    ("discord_webhook", re.compile(r"discord(?:app)?\.com/api/webhooks/[0-9]{16,20}/(?P<v>[A-Za-z0-9_-]{60,})"), ("/api/webhooks/",)),
    ("azure_storage_key", re.compile(r"(?i)AccountKey=(?P<v>[A-Za-z0-9+/]{86}==)"), ("~accountkey=",)),
    ("azure_sas", re.compile(r"(?i)(?<![A-Za-z0-9])sig=(?P<v>[A-Za-z0-9%]{43,})(?=&|[\"'`\s]|$)"), ("~sig=",)),
    ("shopify_token", re.compile(r"(?<![A-Za-z0-9_])shp(?:at|ca|pa|ss)_[0-9a-fA-F]{32}(?![A-Za-z0-9])"), ("shpat_", "shpca_", "shppa_", "shpss_")),
    ("mailgun_key", re.compile(r"(?<![A-Za-z0-9_-])key-[0-9a-f]{32}(?![A-Za-z0-9])"), ("key-",)),
    ("digitalocean_token", re.compile(r"(?<![A-Za-z0-9_])do[opr]_v1_[a-f0-9]{64}(?![A-Za-z0-9])"), ("dop_v1_", "doo_v1_", "dor_v1_")),
    ("hashicorp_token", re.compile(r"(?<![A-Za-z0-9_.-])hv[sbr]\.(?=[A-Za-z_-]*[0-9])[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])"), ("hvs.", "hvb.", "hvr.")),
    ("age_secret_key", re.compile(r"AGE-SECRET-KEY-1[A-Z0-9]{58}"), ("AGE-SECRET-KEY-1",)),
    ("firebase_fcm_key", re.compile(r"(?<![A-Za-z0-9_-])AAAA[A-Za-z0-9_-]{7}:APA91b[A-Za-z0-9_-]{100,}"), (":APA91b",)),
    ("heroku_api_key", re.compile(r"(?i)heroku[\w.\-]{0,20}?\s*[:=]\s*[\"'`]?(?P<v>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"), ("~heroku",)),
]

_CLOUD_PLACEHOLDER = ("EXAMPLE", "XXXXXXXX", "00000000", "12345678", "ABCDEFGH", "ZZZZZZZZ", "AAAAAAAAAAAA", "REDACTED",
                      "PLACEHOLDER", "YOURKEY", "YOUR_KEY", "YOUR-KEY", "DUMMY", "FAKE", "TEST1234", "FOOBAR")


def _cloud_placeholder(v: str) -> bool:
    u = v.upper()
    if any(w in u for w in _CLOUD_PLACEHOLDER):
        return True
    body = re.sub(r"^[A-Za-z]+[_.-]", "", v)
    return bool(_ALL_SAME.match(body)) or _ascending_run(body, 10)


JWT = re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])")

# URL with userinfo; value = password. Group names: u / v / h.
URL_USERINFO = re.compile(
    r"(?i)(?<![A-Za-z0-9+.-])[a-z][a-z0-9+.-]{1,20}://(?P<u>[^\s:/@\"'`<>]{0,64}):(?P<v>[^\s/@\"'`<>]{3,128})@(?P<h>[^\s\"'`/:?#]+)")
# Go database/sql DSN: user:pass@tcp(host:port)/db  or user:pass@unix(/path)/db  or user:pass@/db
GO_DSN = re.compile(r"(?<![A-Za-z0-9_.@:-])(?P<u>[A-Za-z0-9_.-]{1,32}):(?P<v>[^@\s\"'`/]{3,64})@(?:tcp|unix)\(")
# ADO.NET style: ...;Password=xxx;  only counts when the string also looks like a connection string
ADO_PASSWORD = re.compile(r"(?i)(?<![A-Za-z])(?:password|pwd)\s*=\s*(?P<v>[A-Za-z0-9][^;\"'`\s<>{}$%&]{2,127})(?=;|\"|'|`|$|\s)")
ADO_CONTEXT = re.compile(r"(?i)(?:server|data source|host|user id|uid|database|initial catalog|port)\s*=")

_KEYS = (r"pass(?:word|wd|phrase)?|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|secret[_-]?key|"
         r"private[_-]?key|client[_-]?secret|auth[_-]?token|access[_-]?token|refresh[_-]?token|bearer")
# key = "value"   key: "value"   "key": "value"   Key("value") is NOT matched on purpose (needs the quote right after)
_KEY_SUFFIX = r"(?:s|_?(?:value|val|str|string|b64|base64|hex|raw|bytes|data|hash|v\d+|\d+))?"
CRED_ASSIGN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?P<k>[A-Za-z0-9_]{0,32}?(?:" + _KEYS + r")" + _KEY_SUFFIX + r")(?![A-Za-z0-9])[\"']?\s*(?:[:=]|=>|:=)\s*(?:\[\]byte\(|Encoding\.\w+\.GetBytes\(|new\s+\w+\()?(?P<q>[\"'`])(?P<v>[^\s\"'`]{8,})(?P=q)")
# key=value inside URLs / query strings / env-style text
CRED_KV = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?P<k>password|passwd|pwd|secret|api[_-]?key|apikey|access[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|token)=(?P<v>[A-Za-z0-9+/_.~!*-]{8,})(?=[\s&;\"'`)\]]|$)")
# "Authorization": "Basic xxxx" / Bearer xxxx
AUTH_HEADER = re.compile(r"(?i)(?:authorization|auth)[\"']?\s*[:=,]\s*[\"'`]?(?:basic|bearer)\s+(?P<v>[A-Za-z0-9+/=_\-.]{16,})(?![A-Za-z0-9+/=_\-.])")
# Any 8+ char quoted value that we consider a placeholder, not a secret
_PLACEHOLDER_WORDS = (
    "example", "placeholder", "your", "changeme", "change_me", "change-me", "dummy", "fake", "sample", "test",
    "xxx", "***", "...", "secret", "password", "passwd", "redacted", "insert", "replace", "todo", "fixme", "foo",
    "bar", "baz", "qux", "token", "apikey", "api_key", "api-key", "default", "mock", "hunter2", "letmein",
    "123456", "qwerty", "abcdef", "abc123", "admin", "string", "value", "none", "null", "undefined", "unknown",
    "invalid", "wrong", "bad", "correct", "valid", "pass", "mypass", "my_pass", "my-pass", "s3cr", "p@ss",
    "p4ss", "passw0rd", "random", "whatever", "something", "anything", "nothing", "empty", "blank", "deadbeef",
    "cafebabe", "lorem", "ipsum", "hello", "world", "demo", "key", "credential", "env", "getenv",
    "os.", "config", "setting", "option", "name", "label", "title", "message", "description", "header", "basic ",
    "bearer ", "jwt", "oauth", "{{", "${", "<", ">", "%", "$", "\\x", "\\u", "//", "::")
_ALL_SAME = re.compile(r"^(.)\1*$")
_IDENTIFIER_LIKE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_NON_SECRET_KEY = re.compile(r"(?i)page|paging|cursor|continuation|next|prev|csrf|xsrf|antiforgery|cancel|change|"
                             r"publickey|pubkey|public_key|_hash$|hash$|hashed|id$|name$|type$|kind$|path$|file$|url$|uri$")


def weak_credential_value(v: str) -> bool:
    """True for values that are most likely config keys / words / paths rather than secrets."""
    if v[:2].lower() == "0x" and all(c in "0123456789abcdefABCDEF_" for c in v[2:]):
        return True  # hex literal: metadata token / address, not a secret
    if all(c in "0123456789.,_ -" for c in v):
        return True  # amounts, ids, timestamps
    if any(c.isdigit() for c in v):
        return False
    if len(v) >= 20 and any(c.isupper() for c in v) and any(c.islower() for c in v):
        return False
    return True


def looks_placeholder(v: str, key: str = "") -> bool:
    lv = v.lower()
    if any(w in lv for w in _PLACEHOLDER_WORDS):
        return True
    if _ALL_SAME.match(v):
        return True
    if key and key.lower() in lv:
        return True
    # a plain identifier-looking value with no digits is almost never a real secret (e.g. token = "SELECT_STAR")
    if _IDENTIFIER_LIKE.match(v) and not any(c.isdigit() for c in v):
        return True
    if v.startswith(("/", "./", "../", "-", "#", "@", "http://", "https://", "file:")) or " " in v:
        return True
    return False


# high entropy
B64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{32,}={0,2}(?![A-Za-z0-9+/=_-])")
HEX_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:[0-9a-f]{32,128}|[0-9A-F]{32,128})(?![A-Za-z0-9])")
_HASH_LENGTHS = {32, 40, 56, 64, 96, 128}
# base64 of well-known binary formats / DER structures / PEM bodies -> data, not credentials
_BLOB_PREFIXES = ("iVBORw0KGgo",  # PNG
                  "/9j/",  # JPEG
                  "R0lGOD",  # GIF
                  "H4sI",  # gzip
                  "JVBERi0",  # PDF
                  "UEsDB",  # zip
                  "AAAA",  # DER / zero padded
                  "MII",  # DER cert / key / CSR
                  "Qk",  # BMP
                  "PD94bW",  # <?xml
                  "PHN2Zy",  # <svg
                  "PCFET0NUWVBF",  # <!DOCTYPE
                  "d2VibQ",  # webm
                  "AAABAA",  # ico
                  "UklGR",  # RIFF
                  "GkXfo",  # mkv
                  "TVqQ",  # MZ exe
                  "f0VMR",  # ELF
                  "wOFF", "d09GRg",  # woff
                  "0M8R4KGxGuE",  # OLE
                  "SUQz", "//uQ", "//sw",  # mp3
                  "eyJ",  # JSON (jwt is handled separately)
                  "CgoK", "Cg==",  # protobuf descriptors
                  "AQAB", "BAAA", "AAAB",  # RSA e / DER
                  "e30", "e30=",  # {}
                  )
_PEM_BODY_CONTEXT = re.compile(r"-----BEGIN [A-Z0-9 ]+-----")
_URL_HOST_END = re.compile(r"\.[a-z]{2,6}$|\.$")
_CAMEL_WORD = re.compile(r"[A-Z][a-z]{3,}")
_ENTROPY_MIN_DIGITS = 3


def shannon_entropy(s: str) -> float:
    n = len(s)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return -sum((k / n) * math.log2(k / n) for k in counts.values())


def _entropy_candidate(v: str) -> bool:
    if len(v) < 32:
        return False
    if v.startswith(_BLOB_PREFIXES):
        return False
    if v[0] in "/+-_=":
        return False
    digits = sum(c.isdigit() for c in v)
    if digits < _ENTROPY_MIN_DIGITS or digits < len(v) // 10:
        return False
    has_lower = any("a" <= c <= "z" for c in v)
    has_upper = any("A" <= c <= "Z" for c in v)
    if not (has_lower and has_upper):
        return False
    # Go/C# identifiers and snake_case: a long run with '_' and no '+/=' is a code identifier most of the time
    if _IDENTIFIER_LIKE.match(v) and ("_" in v or v[0].isupper() and v[-1].isalpha() and digits < 6):
        return False
    if _ascending_run(v, 8):
        return False
    for seg in re.split(r"[/_-]", v):
        if len(seg) >= 6 and (seg.islower() or seg.isupper()) and seg.isalpha():
            return False  # docs/guard/accuracy..., applyset-v1-..., MY_LONG_CONSTANT_NAME_2
    if len(_CAMEL_WORD.findall(v)) >= 3:
        return False  # NAD1983StatePlaneAlabamaEastFIPS0101: an identifier
    return shannon_entropy(v) > 4.5


def _ascending_run(v: str, n: int) -> bool:
    run = 1
    for i in range(1, len(v)):
        run = run + 1 if ord(v[i]) == ord(v[i - 1]) + 1 else 1
        if run >= n:
            return True
    return False


# e-mail
_EMAIL_LEFT = re.compile(r"[A-Za-z0-9._%+-]{1,64}$")
_EMAIL_RIGHT = re.compile(r"(?P<d>[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}\.)+(?P<tld>[A-Za-z]{2,24}))(?![A-Za-z0-9.-])")


def iter_emails(text: str):
    """Yield (start, end, local, domain) for every e-mail-shaped token; O(number of '@')."""
    i = text.find("@")
    while i != -1:
        lm = _EMAIL_LEFT.search(text, max(0, i - 64), i)
        if lm and (lm.start() == 0 or text[lm.start() - 1] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-"):
            rm = _EMAIL_RIGHT.match(text, i + 1)
            if rm:
                yield lm.start(), rm.end(), text[lm.start():i], rm.group("d")
                i = text.find("@", rm.end())
                continue
        i = text.find("@", i + 1)
_EMAIL_ALLOW_LOCAL = {"noreply", "no-reply", "no_reply", "donotreply", "do-not-reply", "git", "user", "username",
                      "test", "tester", "testuser", "test.user", "foo", "bar", "baz", "admin", "administrator",
                      "root", "someone", "somebody", "you", "me", "example", "john", "jane", "john.doe", "jane.doe",
                      "johndoe", "janedoe", "jdoe", "doe", "alice", "bob", "carol", "dave", "eve", "mallory",
                      "info", "support", "contact", "hello", "hi", "mail", "email", "e-mail", "sales", "security",
                      "webmaster", "postmaster", "hostmaster", "abuse", "help", "team", "dev", "developer",
                      "developers", "devs", "feedback", "service", "services", "office", "name", "first.last",
                      "firstname.lastname", "a", "b", "c", "x", "y", "z", "xxx", "yyy", "zzz", "aaa", "bbb", "user1",
                      "user2", "user3", "test1", "test2", "test3", "nobody", "anonymous", "guest", "customer",
                      "client", "sender", "recipient", "receiver", "to", "from", "cc", "bcc", "reply", "replyto",
                      "placeholder", "dummy", "fake", "sample", "demo", "invalid", "notexist", "nonexistent",
                      "local", "localhost", "mailer-daemon", "bounce", "bounces", "newsletter", "notifications",
                      "notification", "alerts", "alert", "robot", "bot", "ci", "jenkins", "github-actions",
                      "github-actions[bot]", "dependabot[bot]", "renovate[bot]", "41898282+github-actions[bot]",
                      "49699333+dependabot[bot]", "codeowners", "owner", "owners", "maintainer", "maintainers",
                      "author", "authors", "committer", "me@here", "nobody@nowhere", "a.b", "ab", "abc", "xyz"}
_EMAIL_ALLOW_DOMAIN_SUFFIX = ("example.com", "example.org", "example.net", "example.io", "example.co", "example.edu",
                              "example.gov", "example.uk", "localhost", "localdomain", "local", "test", "example",
                              "invalid", "localhost.com", "domain.com", "domain.org", "domain.net", "email.com",
                              "email.org", "mail.com", "test.com", "test.org", "test.net", "test.io", "test.local",
                              "foo.com", "bar.com", "baz.com", "foo.org", "foo.bar", "foo.net", "foo.io",
                              "company.com", "company.org", "corp.com", "acme.com", "acme.org", "acme.io", "site.com",
                              "website.com", "server.com", "host.com", "hostname.com", "yourdomain.com",
                              "your-domain.com", "mydomain.com", "my-domain.com", "mycompany.com", "mysite.com",
                              "myserver.com", "somewhere.com", "somedomain.com", "something.com", "anything.com",
                              "nowhere.com", "sample.com", "dummy.com", "fake.com", "placeholder.com", "contoso.com",
                              "fabrikam.com", "adventure-works.com", "northwind.com", "tailspintoys.com",
                              "yourcompany.com", "mail.example", "xyz.com", "abc.com", "aaa.com", "bbb.com",
                              "tempuri.org", "w3.org", "schemas.microsoft.com", "googlegroups.com",
                              "users.noreply.github.com", "noreply.github.com", "lists.sourceforge.net",
                              "openssh.com", "libssh.org", "ssh.com", "odata.count", "odata.context", "odata.type",
                              "odata.id", "odata.etag", "odata.nextlink", "odata.deltalink", "odata.bind",
                              "evil.com", "attacker.com", "malicious.com", "bad.com", "good.com", "some-host.com",
                              "somehost.com", "musterma.nn", "mustermann.de", "a.com", "b.com", "c.com", "a.b.c",
                              "a.b", "x.y", "x.y.z", "foo.bar.baz", "my.com", "me.com", "here.com", "there.com",
                              "nowhere.org", "whatever.com", "random.com", "somewhere.org", "local.host", "internal",
                              "lan", "home", "corp", "intranet", "private", "localnet", "home.arpa", "in-addr.arpa",
                              "onion", "i2p")
# licence / author header lines keep their e-mail (public attribution, model should learn the shape)
_HEADER_LINE = re.compile(r"(?i)copyright|\(c\)|©|@author|\bauthors?\b|maintainers?|contributors?|written by|"
                          r"created by|developed by|\blicen[cs]e|contact|\bby\s+[A-Z][a-z]+ [A-Z]|signed-off-by|co-authored-by")


_FILE_EXT_TLDS = {"png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "jar", "js", "ts", "jsx", "tsx", "go", "cs",
                  "css", "scss", "html", "htm", "json", "xml", "yaml", "yml", "txt", "md", "zip", "gz", "tar", "vdex",
                  "dex", "apk", "aab", "so", "dll", "exe", "pdf", "mp3", "mp4", "wav", "ttf", "otf", "woff", "woff2",
                  "class", "proto", "pb", "wasm", "bin", "dat", "db", "sqlite", "csv", "tsv", "log", "cfg", "ini",
                  "toml", "sh", "bat", "ps1", "py", "rb", "rs", "java", "kt", "swift", "c", "h", "cpp", "hpp", "tmpl",
                  "tpl", "mustache", "hbs", "ejs", "vue", "svelte", "map", "lock", "sum", "mod", "d"}


_CS_KEYWORDS = {"internal", "event", "namespace", "class", "string", "object", "params", "base", "this", "default",
                "operator", "checked", "unchecked", "lock", "out", "ref", "in", "is", "as", "new", "null", "true", "false",
                "void", "var", "int", "long", "bool", "byte", "char", "double", "float", "decimal", "short", "uint",
                "ulong", "ushort", "sbyte", "static", "public", "private", "protected", "abstract", "sealed", "virtual",
                "override", "readonly", "const", "struct", "enum", "interface", "delegate", "typeof", "sizeof", "fixed",
                "unsafe", "using", "return", "yield", "await", "async", "where", "select", "from", "group", "into",
                "orderby", "join", "let", "on", "equals", "by", "ascending", "descending", "value", "get", "set", "add",
                "remove", "partial", "global", "alias", "dynamic", "nameof", "when", "else", "if", "for", "foreach",
                "while", "do", "switch", "case", "break", "continue", "goto", "try", "catch", "finally", "throw",
                "implicit", "explicit", "extern", "volatile", "stackalloc", "func", "type", "map", "chan", "range"}


def email_allowed(local: str, domain: str, line_text: str) -> bool:
    ll = local.lower()
    dl = domain.lower()
    labels = dl.split(".")
    if labels[-1] in _FILE_EXT_TLDS:
        return True  # icon@3x.png, foo@classes.jar: file names, not addresses
    if labels[0] in _CS_KEYWORDS:
        return True  # C# verbatim identifiers: Foo.@internal.Bar, x.@event.Add
    if len(labels[0]) <= 2 or len(ll) <= 1:
        return True  # a@x.com, u@e.com, t@t.tt: test placeholders
    if ll in _EMAIL_ALLOW_LOCAL or ll.startswith(("noreply", "no-reply", "no_reply", "donotreply", "test", "user",
                                                   "example", "foo", "bar", "dummy", "fake", "sample", "someone",
                                                   "somebody", "invalid", "nobody", "placeholder", "mock")):
        return True
    for suf in _EMAIL_ALLOW_DOMAIN_SUFFIX:
        if dl == suf or dl.endswith("." + suf):
            return True
    if dl.startswith(("test", "example", "fake", "dummy", "sample", "mock", "my", "some", "foo", "bar", "invalid",
                      "placeholder", "localhost", "demo")):
        return True
    if _HEADER_LINE.search(line_text):
        return True
    return False


# IPv4
IPV4 = re.compile(r"(?<![\d.\w])(?P<a>\d{1,3})\.(?P<b>\d{1,3})\.(?P<c>\d{1,3})\.(?P<d>\d{1,3})(?![\d.\w])")
_WELLKNOWN_IPS = {"1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9", "149.112.112.112", "4.4.4.4", "4.2.2.2",
                  "1.2.3.4", "5.6.7.8", "9.10.11.12", "4.3.2.1", "1.2.3.5", "1.2.3.6", "2.3.4.5", "6.7.8.9",
                  "208.67.222.222", "208.67.220.220", "1.1.1.2", "1.0.0.2", "1.1.1.3", "1.0.0.3", "11.22.33.44",
                  "123.123.123.123", "111.111.111.111", "222.222.222.222", "12.34.56.78", "98.76.54.32",
                  "1.1.1.0", "1.2.3.0", "2.2.2.2", "3.3.3.3", "5.5.5.5", "6.6.6.6", "7.7.7.7", "11.11.11.11",
                  "12.12.12.12", "100.100.100.100", "200.200.200.200", "201.201.201.201", "202.202.202.202",
                  "99.99.99.99", "123.45.67.89", "12.345.67.89", "1.2.3.44", "1.2.4.8", "1.3.5.7", "2.4.6.8",
                  "209.85.128.1", "93.184.216.34", "93.184.215.14", "216.58.192.0", "142.250.0.0", "104.16.0.0",
                  "17.17.17.17", "13.13.13.13", "14.14.14.14", "15.15.15.15", "16.16.16.16", "18.18.18.18",
                  "19.19.19.19", "20.20.20.20", "21.21.21.21", "22.22.22.22", "23.23.23.23", "24.24.24.24",
                  "25.25.25.25", "33.33.33.33", "44.44.44.44", "55.55.55.55", "66.66.66.66", "77.77.77.77",
                  "88.88.88.88", "1.2.3.4", "4.5.6.7", "5.5.5.5", "10.11.12.13", "11.12.13.14", "13.14.15.16",
                  "20.30.40.50", "50.60.70.80", "100.101.102.103", "123.234.123.234", "119.29.29.29", "223.5.5.5",
                  "223.6.6.6", "114.114.114.114", "180.76.76.76", "210.2.4.8", "1.2.4.8", "101.226.4.6", "8.26.56.26",
                  "8.20.247.20", "64.6.64.6", "64.6.65.6", "77.88.8.8", "77.88.8.1", "76.76.2.0", "76.76.19.19",
                  "94.140.14.14", "94.140.15.15", "185.228.168.9", "185.228.169.9", "156.154.70.1", "156.154.71.1",
                  "140.82.112.3", "140.82.112.4", "140.82.113.3", "140.82.113.4", "140.82.114.3", "140.82.114.4",
                  "104.16.0.1", "104.16.1.1", "151.101.1.69", "34.117.59.81"}
_IP_CONTEXT_VERSION = re.compile(r"(?i)version|assembly|\bver\b|v\d|release|\bbuild\b|section|§|chapter|spec|rfc|iso|pdf")
_IP_CONTEXT_NET = re.compile(r"(?i)\bip|addr|host|dns|server|route|peer|gateway|cidr|subnet|netmask|resolver|"
                             r"nameserver|ipv4|inet|sockaddr|x-real|x-forwarded|listen|dial|nmap|whois|ping")


_COMMENT_LINE = re.compile(r"^\s*(?://|/\*|\*|#|///|--)")


def ip_class(a: int, b: int, c: int, d: int, line_text: str = "", anchored: bool = False) -> str | None:
    """None -> not interesting (private, reserved, version-like).  'wellknown' or 'public'.
    anchored: the literal is quoted or carries a :port / CIDR suffix (strong evidence it is an address)."""
    if a > 255 or b > 255 or c > 255 or d > 255:
        return None
    if a == 0 or a == 10 or a == 127 or a >= 224:
        return None
    if a == 172 and 16 <= b <= 31:
        return None
    if a == 192 and (b == 168 or (b == 0 and c in (0, 2)) or (b == 88 and c == 99)):
        return None
    if a == 169 and b == 254:
        return None
    if a == 100 and 64 <= b <= 127:
        return None
    if a == 198 and (b in (18, 19) or (b == 51 and c == 100)):
        return None
    if a == 203 and b == 0 and c == 113:
        return None
    if a == 255:
        return None
    # version-like: x.y.0.0, x.0.0.y, 1.0.0.0, or small octets near a version keyword
    if (c == 0 and d == 0) or (b == 0 and c == 0):
        return None
    if a <= 20 and b <= 20:
        net = _IP_CONTEXT_NET.search(line_text)
        if _IP_CONTEXT_VERSION.search(line_text) and not net:
            return None  # Version: "3.14.1.42"
        if c <= 20 and d <= 20 and not net:
            return None  # "2.5.1.3", 3.3.5.2: versions and section numbers; real addresses this small are the well-known resolvers
        if a < 10 and not net and _COMMENT_LINE.match(line_text):
            return None  # // e.g. "4.1.0.123" -> "4.1.0"
    if sum(x >= 10 for x in (a, b, c, d)) < 3 and not anchored and not _IP_CONTEXT_NET.search(line_text):
        return None  # 12.7.4.3 / 12.6.4.12-style section and version numbers in prose
    if f"{a}.{b}.{c}.{d}" in _WELLKNOWN_IPS:
        return "wellknown"
    return "public"


# phone-like (report only)
PHONE = re.compile(r"(?<![\w.+-])(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w.-])|(?<![\w.+-])\+\d{10,14}(?![\w.-])")


# --------------------------------------------------------------------------------------------------------------
# detection

def _line_starts(text: str) -> list[int]:
    starts = [0]
    i = text.find("\n")
    while i != -1:
        starts.append(i + 1)
        i = text.find("\n", i + 1)
    return starts


def _line_of(starts: list[int], pos: int) -> int:
    import bisect
    return bisect.bisect_right(starts, pos)  # 1-based


def _line_text(text: str, starts: list[int], ln: int) -> str:
    s = starts[ln - 1]
    e = starts[ln] - 1 if ln < len(starts) else len(text)
    return text[s:e]


def find_matches(text: str, path: str = "", with_report_only: bool = True) -> list[Match]:
    out: list[Match] = []
    starts: list[int] | None = None
    tl = text.lower()

    def lstarts() -> list[int]:
        nonlocal starts
        if starts is None:
            starts = _line_starts(text)
        return starts

    # private keys
    for m in (PRIVATE_KEY.finditer(text) if "PRIVATE KEY" in text else ()):
        end = text.find("-----END ", m.end(), m.end() + 65536)
        if end != -1:
            e2 = text.find("-----", end + 9, end + 80)
            end = e2 + 5 if e2 != -1 else end + 9
        else:
            end = m.end()
        body = text[m.end():end]
        body = body[: body.find("-----END ")] if "-----END " in body else body
        if not _PEM_BODY.fullmatch(body) or len(re.sub(r"[^A-Za-z0-9+/=]", "", body)) < 64:
            continue  # "-----BEGIN RSA PRIVATE KEY-----\n(cert contents)\n..." : documentation placeholder
        out.append(Match("private_key", m.group(0)[len("-----BEGIN "):-5].strip().lower(), m.start(), end))

    # cloud keys
    for sub, rx, needles in CLOUD_KEYS:
        if not any((n[1:] in tl) if n[0] == "~" else (n in text) for n in needles):
            continue
        for m in rx.finditer(text):
            g = "v" if "v" in rx.groupindex else 0
            if _cloud_placeholder(m.group(g)):
                continue  # AKIAIOSFODNN7EXAMPLE, ghp_xxxxxxxx..., 0000000...: documentation values
            out.append(Match("cloud_key", sub, m.start(g), m.end(g)))

    for m in (JWT.finditer(text) if "eyJ" in text else ()):
        out.append(Match("jwt", "jwt", m.start(), m.end()))

    # connection strings
    has_at = "@" in text
    for m in (URL_USERINFO.finditer(text) if has_at and "://" in text else ()):
        u, p, h = m.group("u"), m.group("v"), m.group("h")
        if looks_placeholder(p) or p.lower() == u.lower() or any(c in p for c in "[](){}\\|*?"):
            continue
        if u.lower() in ("user", "username", "usr", "login") and p.lower() in ("pass", "pwd", "password"):
            continue
        local = h.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "db", "mysql", "postgres", "redis", "mongo", "host",
                              "hostname", "example.com", "localhost.localdomain") or h.lower().endswith((".example.com", ".local", ".test"))
        out.append(Match("conn_string", "url_local" if local else "url", m.start("v"), m.end("v"), extra={"user": u, "host": h}))
    for m in (GO_DSN.finditer(text) if "@tcp(" in text or "@unix(" in text else ()):
        u, p = m.group("u"), m.group("v")
        if looks_placeholder(p) or p.lower() == u.lower():
            continue
        out.append(Match("conn_string", "go_dsn", m.start("v"), m.end("v"), extra={"user": u}))
    has_pw = "password" in tl or "pwd" in tl
    for m in (ADO_PASSWORD.finditer(text) if has_pw else ()):
        ln = _line_of(lstarts(), m.start())
        lt = _line_text(text, lstarts(), ln)
        if not ADO_CONTEXT.search(lt):
            continue
        if looks_placeholder(m.group("v")):
            continue
        out.append(Match("conn_string", "ado", m.start("v"), m.end("v")))

    # generic credentials
    has_cred = has_pw or any(k in tl for k in ("pass", "secret", "token", "key", "bearer", "credential"))
    for m in (CRED_ASSIGN.finditer(text) if has_cred else ()):
        k, v = m.group("k"), m.group("v")
        if looks_placeholder(v, k) or weak_credential_value(v) or _NON_SECRET_KEY.search(k):
            continue  # PublicKeyToken=..., page_token / cursor, token_hash, token_id: not secrets
        out.append(Match("credential", "assign", m.start("v"), m.end("v"), extra={"key": k}))
    for m in (CRED_KV.finditer(text) if has_cred else ()):
        k, v = m.group("k"), m.group("v")
        if looks_placeholder(v, k) or weak_credential_value(v) or text[max(0, m.start() - 9):m.start()].lower().endswith("publickey"):
            continue
        if _NON_SECRET_KEY.search(text[max(0, m.start("k") - 12):m.start("k")] + k):
            continue  # ...?page_token=..., next_token=
        out.append(Match("credential", "kv", m.start("v"), m.end("v"), extra={"key": k}))
    for m in (AUTH_HEADER.finditer(text) if "basic " in tl or "bearer " in tl else ()):
        v = m.group("v")
        if looks_placeholder(v):
            continue
        out.append(Match("credential", "auth_header", m.start("v"), m.end("v")))

    # high entropy base64-ish literals
    for m in B64_TOKEN.finditer(text):
        v = m.group(0)
        if not _entropy_candidate(v):
            continue
        # skip PEM bodies (certificates, public keys, CSRs) - the surrounding header tells
        pre = text.rfind("-----BEGIN ", max(0, m.start() - 4000), m.start())
        if pre != -1 and text.find("-----END ", pre, m.start()) == -1:
            continue
        # skip data: URIs and URL paths (github.com/owner/repo/blob/<sha>/...)
        pre12 = text[max(0, m.start() - 12):m.start()]
        if pre12.endswith("base64,") or "://" in pre12 or _URL_HOST_END.search(pre12):
            continue
        out.append(Match("high_entropy", "b64_%d" % min(len(v) // 32 * 32, 256), m.start(), m.end()))

    if with_report_only:
        for m in HEX_TOKEN.finditer(text):
            n = m.end() - m.start()
            if n in _HASH_LENGTHS:
                out.append(Match("hash_like", "hex%d" % n, m.start(), m.end()))

    # e-mails
    for st, en, local, dom in iter_emails(text):
        if text[en:en + 1] in ("(", "[") or local.endswith(".") or local.startswith("."):
            continue  # method call in a comment; C# namespace escape: com.espertech.esper.common.@internal.util
        if text[st - 1:st] == ":" and "://" in text[max(0, st - 80):st]:
            continue  # URL userinfo https://user:token@host - the conn_string category owns this
        ln = _line_of(lstarts(), st)
        lt = _line_text(text, lstarts(), ln)
        if email_allowed(local, dom, lt):
            if with_report_only:
                out.append(Match("email_allow", "allow", st, en))
        else:
            out.append(Match("email", "email", st, en, extra={"domain": dom.lower()}))

    # IPv4
    for m in IPV4.finditer(text):
        a, b, c, d = (int(m.group(x)) for x in "abcd")
        ln = _line_of(lstarts(), m.start())
        lt = _line_text(text, lstarts(), ln)
        before, after = text[m.start() - 1:m.start()], text[m.end():m.end() + 2]
        anchored = before in ("\"", "'", "`") or (after[:1] in (":", "/") and after[1:2].isdigit())
        cls = ip_class(a, b, c, d, lt, anchored)
        if cls == "public":
            out.append(Match("public_ip", "ip", m.start(), m.end(), extra={"net": f"{a}.{b}"}))
        elif cls == "wellknown" and with_report_only:
            out.append(Match("ip_wellknown", "wellknown", m.start(), m.end()))

    if with_report_only:
        for m in PHONE.finditer(text):
            out.append(Match("phone", "phone", m.start(), m.end()))

    if not out:
        return out
    # resolve overlaps: keep the higher-priority (lower PRIORITY index) match, then the earlier one
    out.sort(key=lambda x: (x.start, PRIORITY[x.category], -(x.end - x.start)))
    kept: list[Match] = []
    for mt in out:
        if kept and mt.start < kept[-1].end:
            prev = kept[-1]
            if PRIORITY[mt.category] < PRIORITY[prev.category]:
                kept[-1] = mt
            continue
        kept.append(mt)
    st = lstarts()
    for mt in kept:
        mt.line = _line_of(st, mt.start)
    return kept


def classify_line(line: str) -> str | None:
    """Highest-priority category present in one line (None if clean).  Report-only categories count too."""
    ms = find_matches(line, with_report_only=True)
    if not ms:
        return None
    return min(ms, key=lambda m: PRIORITY[m.category]).category


# --------------------------------------------------------------------------------------------------------------
# redaction for reports

def mask(m: Match, text: str) -> str:
    v = m.value(text)
    n = len(v)
    if m.category in ("email", "email_allow"):
        local, _, dom = v.partition("@")
        parts = dom.split(".")
        dom_m = parts[0][:1] + "*" * max(1, len(parts[0]) - 1) + "." + ".".join(parts[1:]) if len(parts) > 1 else "*"
        return local[:1] + "*" * max(1, min(len(local) - 1, 8)) + "@" + dom_m
    if m.category in ("public_ip", "ip_wellknown"):
        a, b, c, d = v.split(".")
        return f"{a}.{b}.*.*"
    if m.category == "phone":
        return "+" * v.startswith("+") + "*" * (n - v.startswith("+") - 2) + v[-2:]
    if m.category == "jwt":
        return "eyJ***.eyJ***.***"
    if m.category == "private_key":
        return v
    if m.category == "hash_like":
        return v[:6] + "*" * min(n - 6, 10) + f"(hex{n})"
    keep = 4 if m.category in ("cloud_key", "high_entropy") else 2
    return v[:keep] + "*" * min(n - keep, 16) + (f"(len {n})" if n > keep + 16 else "")


def masked_line(text: str, matches: list[Match], line_no: int, width: int = 160) -> str:
    """The source line with every matched value replaced by its mask (for human review of samples)."""
    starts = _line_starts(text)
    s = starts[line_no - 1]
    e = starts[line_no] - 1 if line_no < len(starts) else len(text)
    line = text[s:e]
    ms = [m for m in matches if m.line == line_no]
    for m in sorted(ms, key=lambda x: -x.start):
        line = line[: m.start - s] + "[" + m.category.upper() + ":" + mask(m, text) + "]" + line[m.end - s:]
    line = line.strip()
    return line if len(line) <= width else line[: width - 1] + "…"


# --------------------------------------------------------------------------------------------------------------
# scrubbing policy

DEFAULT_POLICY = {
    "drop_on_private_key": True,     # (a) file with a PRIVATE KEY block is dropped
    "drop_on_credential_hits": 3,    # (a) >= N hits in CREDENTIAL_CATEGORIES -> drop file
    "drop_on_scrub_matches": 100,    # a file with >= N scrubbable matches is a data file (IP list, base64 dump), not code
    "scrub": sorted(SCRUBBED),       # (b-d) categories whose VALUE is replaced by PLACEHOLDER
}


def _new_stats() -> dict:
    return {"dropped": None, "lines_scrubbed": 0, "bytes_affected": 0, "matches": {}}


def scrub(text: str, path: str = "", policy: dict = DEFAULT_POLICY) -> tuple[str | None, dict]:
    """Return (clean_text, stats).  clean_text is None when the file must be dropped.
    stats: dropped (None | 'private_key' | 'credential_hits' | 'data_file'), lines_scrubbed, bytes_affected (chars of the
    original spans replaced), matches {category: count}."""
    ms = find_matches(text, path, with_report_only=False)
    return scrub_matches(text, ms, policy)


def scrub_matches(text: str, ms: list[Match], policy: dict = DEFAULT_POLICY) -> tuple[str | None, dict]:
    """Apply the policy to an existing match list (report-only categories are ignored)."""
    stats = _new_stats()
    ms = [m for m in ms if m.category in SCRUBBED]
    if not ms:
        return text, stats
    for m in ms:
        stats["matches"][m.category] = stats["matches"].get(m.category, 0) + 1
    if policy.get("drop_on_private_key") and stats["matches"].get("private_key"):
        stats["dropped"] = "private_key"
        stats["bytes_affected"] = len(text)
        return None, stats
    n_cred = sum(stats["matches"].get(c, 0) for c in CREDENTIAL_CATEGORIES)
    thr = policy.get("drop_on_credential_hits")
    if thr and n_cred >= thr:
        stats["dropped"] = "credential_hits"
        stats["bytes_affected"] = len(text)
        return None, stats
    scrub_cats = set(policy.get("scrub", ()))
    thr = policy.get("drop_on_scrub_matches")
    if thr and sum(1 for m in ms if m.category in scrub_cats) >= thr:
        stats["dropped"] = "data_file"
        stats["bytes_affected"] = len(text)
        return None, stats
    pieces = []
    pos = 0
    lines = set()
    for m in ms:  # already sorted & non-overlapping
        if m.category not in scrub_cats:
            continue
        pieces.append(text[pos:m.start])
        pieces.append(PLACEHOLDER[m.category])
        stats["bytes_affected"] += m.end - m.start
        lines.add(m.line)
        pos = m.end
    pieces.append(text[pos:])
    stats["lines_scrubbed"] = len(lines)
    return "".join(pieces), stats


def scrub_bytes(data: bytes, path: str = "", policy: dict = DEFAULT_POLICY) -> tuple[bytes | None, dict]:
    """bytes wrapper for encode_corpus.py: `data = scrub_bytes(read_file(lang, d), d['path'])[0]; if data is None: skip`."""
    # cheap pre-check: nothing to do for files without any trigger byte sequences
    text = data.decode("utf-8", errors="surrogateescape")
    clean, stats = scrub(text, path, policy)
    if clean is None:
        return None, stats
    if clean is text:
        return data, stats
    return clean.encode("utf-8", errors="surrogateescape"), stats
