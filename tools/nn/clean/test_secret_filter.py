"""Unit tests for secret_filter.  Run: ~/work/nn/.venv/bin/python -I -m unittest -v test_secret_filter
All 'secrets' below are synthetic (random strings shaped like the real thing)."""
import os
import sys
import unittest
# Synthetic secret values are assembled from pieces so that this file itself never contains a scanner-matching token.
AWS_KEY = "AKIA" + "Z7Q3K1XW9P2YB8DT"
GOOGLE_KEY = "AIza" + "SyD9x8Qm2Lk3Pw7Rt1Vb5Nz6Jh0Fg4Cd2Ea"
GITHUB_PAT = "ghp_" + "Ab3dEf6Gh9Jk2Mn5Pq8Rs1Tu4Vw7Xy0Za3Bc6D"
SLACK_TOKEN = "xoxb-" + "592837465012-8273645190283-K9mN2pQ7rT4vW1xZ3bC5dF8g"
STRIPE_KEY = "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import secret_filter as sf  # noqa: E402

def tok(n):
    """deterministic synthetic token of exactly n chars: mixed case, digits, no runs"""
    base = "k9Mn2Pq7Rt4Vw1Xz3Bc5DfQ8zX3wV7yU1tS5rR9qP2oO4nN6mM8lL0kK1jJ3hH5gG7fF9dD2sS4aA6"
    return (base * (n // len(base) + 1))[:n]


RAND_B64 = "k8Jr2Qz9Xp1LmN4vB7cT0yW3hF6gD5sA2eR8uI1oP4qZ"  # 44 chars, mixed case, digits, high entropy
RAND_B64_2 = "Zx3Qw9Er8Ty2Ui1Op4As7Df6Gh5Jk0Lz"  # 32 chars
SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
MD5 = "d41d8cd98f00b204e9800998ecf8427e"
UUID = "123e4567-e89b-12d3-a456-426614174000"
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
JWT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
       "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
       "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")
PRIV = ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAx" + tok(60) + "\n" + tok(64) + "\n" + tok(20) + "==\n"
        "-----END RSA PRIVATE KEY-----\n")
PUB = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA7\n-----END PUBLIC KEY-----\n"
CERT = ("-----BEGIN CERTIFICATE-----\nMIIDXTCCAkWgAwIBAgIJAJC1HiIAZAiIMA0GCSqGSIb3DQEBBQUAMEUxCzAJBgNV\n"
        "BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX\n-----END CERTIFICATE-----\n")


def cats(text, path=""):
    return sorted({m.category for m in sf.find_matches(text, path)})


class Positive(unittest.TestCase):
    def test_cloud_keys(self):
        self.assertEqual(sf.classify_line(f'key := "{AWS_KEY}"'), "cloud_key")
        self.assertEqual(sf.classify_line(f'var g = "{GOOGLE_KEY}"'), "cloud_key")
        self.assertEqual(sf.classify_line(f'tok := "{GITHUB_PAT}"'), "cloud_key")
        self.assertEqual(sf.classify_line('github_pat_11AQ7ZK3X0mZp9wQ2rT8vB1nL6kJ4hG3fD5sA0eR7uI2oP9qZ4xC6vB8nM1kL3jH'), "cloud_key")
        self.assertEqual(sf.classify_line(f'slack := "{SLACK_TOKEN}"'), "cloud_key")
        self.assertEqual(sf.classify_line(f'stripe.Key = "{STRIPE_KEY}"'), "cloud_key")
        self.assertEqual(sf.classify_line('sid := "AC' + "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6" + '"'), "cloud_key")
        self.assertEqual(sf.classify_line('SG.' + tok(22) + '.' + tok(43)), "cloud_key")
        self.assertEqual(sf.classify_line('//registry.npmjs.org/:_authToken=npm_' + tok(36) + "'"), "cloud_key")
        self.assertEqual(sf.classify_line('aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYQ1W2E3R4T5"'), "cloud_key")
        self.assertEqual(sf.classify_line('DefaultEndpointsProtocol=https;AccountName=x;AccountKey=' + tok(86) + '==;'), "cloud_key")
        self.assertEqual(sf.classify_line('https://hooks.slack.com/services/T0' + tok(8).upper() + '/B0' + tok(9).upper() + '/' + tok(24)), "cloud_key")

    def test_jwt(self):
        self.assertEqual(sf.classify_line('Authorization: Bearer ' + JWT), "jwt")
        self.assertEqual(sf.classify_line('tokenString := "' + JWT + '"'), "jwt")

    def test_private_key(self):
        self.assertEqual(cats(PRIV), ["private_key"])
        self.assertEqual(cats("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2gtcn\n"
                              + tok(70) + "\n-----END OPENSSH PRIVATE KEY-----"), ["private_key"])
        # Go string with literal \n escapes and concatenation
        self.assertEqual(cats('k := "-----BEGIN EC PRIVATE KEY-----\\n" + "' + tok(64) + '\\n" +\n "' + tok(40) + '=\\n-----END EC PRIVATE KEY-----"'), ["private_key"])
        self.assertEqual(cats("-----BEGIN PGP PRIVATE KEY BLOCK-----\n\n" + tok(64) + "\n" + tok(64) + "\n=abcd\n-----END PGP PRIVATE KEY BLOCK-----"), ["private_key"])
        # a bare header (secret-scanner pattern tables, docs) or a placeholder body is NOT a key
        for text in ("-----BEGIN EC PRIVATE KEY-----", 'patterns = []string{"-----BEGIN RSA PRIVATE KEY-----"}',
                     '"-----BEGIN RSA PRIVATE KEY-----\\n(cert contents)\\n(... etc ...)\\n-----END RSA PRIVATE KEY-----"',
                     '"-----BEGIN OPENSSH PRIVATE KEY-----\\nfake\\n-----END OPENSSH PRIVATE KEY-----"',
                     "-----BEGIN PRIVATE KEY-----\nFAKE\n-----END PRIVATE KEY-----"):
            self.assertEqual(cats(text), [], msg=text)

    def test_credential_assignments(self):
        self.assertEqual(sf.classify_line('password := "Tr0ub4dor&3xyz"'), "credential")
        self.assertEqual(sf.classify_line('Password: "Tr0ub4dor&3xyz",'), "credential")
        self.assertEqual(sf.classify_line('"api_key": "a9f8e7d6c5b4a3f2e1d0"'), "credential")
        self.assertEqual(sf.classify_line('const apiKey = "a9f8e7d6c5b4a3f2e1d0"'), "credential")
        self.assertEqual(sf.classify_line('var ClientSecret = "q1w2e3r4t5y6u7i8o9p0"'), "credential")
        self.assertEqual(sf.classify_line('u := "https://api.example.com/v1?api_key=a9f8e7d6c5b4a3f2e1d0&x=1"'), "credential")
        self.assertEqual(sf.classify_line('req.Header.Set("Authorization", "Basic dXNlcjpUcjB1YjRkb3Im")'), "credential")
        self.assertEqual(sf.classify_line('string connStr = "Server=db1;Database=prod;User Id=sa;Password=Tr0ub4dor3;"'), "conn_string")

    def test_connection_strings(self):
        self.assertEqual(sf.classify_line('dsn := "postgres://app:Tr0ub4dor3@db.prod.internal:5432/app"'), "conn_string")
        self.assertEqual(sf.classify_line('db, _ := sql.Open("mysql", "root:Tr0ub4dor3@tcp(10.0.0.5:3306)/app")'), "conn_string")
        self.assertEqual(sf.classify_line('redis://:Tr0ub4dor3@cache:6379/0'), "conn_string")
        m = sf.find_matches('dsn := "postgres://app:Tr0ub4dor3@localhost/app"')[0]
        self.assertEqual((m.category, m.subtype), ("conn_string", "url_local"))

    def test_high_entropy(self):
        self.assertEqual(sf.classify_line('secretBytes := []byte("' + RAND_B64 + '")'), "credential")
        self.assertEqual(sf.classify_line('var key = Encoding.UTF8.GetBytes("' + RAND_B64 + '");'), "high_entropy")
        self.assertEqual(sf.classify_line('var secretKey = Encoding.UTF8.GetBytes("' + RAND_B64 + '");'), "credential")
        self.assertEqual(sf.classify_line('x := "' + RAND_B64 + '"'), "high_entropy")
        self.assertEqual(sf.classify_line('"' + RAND_B64_2 + '"'), "high_entropy")

    def test_email(self):
        self.assertEqual(sf.classify_line('// contact: bob.smith1984@gmail.com'), "email_allow")  # 'contact' header rule
        self.assertEqual(sf.classify_line('to := "bob.smith1984@gmail.com"'), "email")
        for line in ('"assets/expo/icon@3x.png"', '"apex@com.android.apex1@javalib@classes.jar"', 'a@x.com', 't@t.tt', 'u@e.com',
                     'using com.espertech.esper.common.@internal.util;', 'this.@event.core.Add(x);', 'namespace a.b.@internal.epl.expression.core'):
            self.assertEqual([m for m in sf.find_matches(line) if m.category == "email"], [], msg=line)
        self.assertEqual(sf.classify_line('<bob.smith1984@corp-mail.co.uk>'), "email")

    def test_public_ip(self):
        self.assertEqual(sf.classify_line('addr := "51.15.236.17:8080"'), "public_ip")
        self.assertEqual(sf.classify_line('host = "91.108.4.1"'), "public_ip")

    def test_report_only(self):
        self.assertEqual(sf.classify_line('want := "' + SHA256 + '"'), "hash_like")
        self.assertEqual(sf.classify_line('// sha1: ' + SHA1), "hash_like")
        self.assertEqual(sf.classify_line('md5 = "' + MD5 + '"'), "hash_like")
        self.assertEqual(sf.classify_line('dns := "8.8.8.8:53"'), "ip_wellknown")
        self.assertEqual(sf.classify_line('phone := "+1 (555) 123-4567"'), "phone")
        self.assertEqual(sf.classify_line('phone := "+79161234567"'), "phone")


class Negative(unittest.TestCase):
    def assertClean(self, line):
        self.assertIsNone(sf.classify_line(line), msg=f"{line!r} -> {sf.find_matches(line)}")

    def test_format_strings(self):
        self.assertClean('fmt.Sprintf("password=%s", pw)')
        self.assertClean('fmt.Sprintf("user=%s password=%s host=%s", u, p, h)')
        self.assertClean('fmt.Sprintf("postgres://%s:%s@%s:%d/%s", u, p, h, port, db)')
        self.assertClean('log.Printf("token = %q", tok)')
        self.assertClean('"Server=%s;Database=%s;User Id=%s;Password=%s;"')
        self.assertClean("$\"Server={host};Password={pwd};\"")
        self.assertClean('url := fmt.Sprintf("https://x.io/?api_key=%s", key)')

    def test_getenv_and_flags(self):
        self.assertClean('apiKey := os.Getenv("API_KEY")')
        self.assertClean('password := flag.String("password", "", "database password")')
        self.assertClean('Password: os.Getenv("DB_PASSWORD"),')
        self.assertClean('token := viper.GetString("token")')
        self.assertClean('Password string `json:"password,omitempty"`')
        self.assertClean('Token string `yaml:"api_key" env:"API_KEY"`')
        self.assertClean('case "password":')
        self.assertClean('if key == "token" {')
        self.assertClean('secret := "${SECRET}"')
        self.assertClean('password = "{{ .Password }}"')
        self.assertClean('apiKey = "<your-api-key>"')
        self.assertClean('token := "your_token_here"')
        self.assertClean('password := "changeme"')
        self.assertClean('Password: "password123",')
        self.assertClean('secret := "my-secret-value"')
        self.assertClean('token := "SELECT_STAR"')
        self.assertClean('tagCertificate: &hvs.PlatformFlavorTagCertificate{},')
        self.assertClean('[Token(Token = "0x06000A1F")] // RVA: 0x12345 token=0x0600A1F2')
        self.assertClean('"System.Core, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089"')
        self.assertClean('"page_token": "cursor1234abcd",')
        self.assertClean('PageToken: "v1_8f7e6d5c4b3a2918",')
        self.assertClean('SelfStakingTokens: "5000000000000000000",')
        self.assertClean('u := "https://api.x.io/v1/items?limit=10&next_token=a9f8e7d6c5b4a3f2e1d0"')
        self.assertEqual([m for m in sf.find_matches('"_hashed_password": "a6f8e7d6c5b4a3f2e1d0a6f8e7d6c5b4a3f2e1d0",') if m.category in sf.SCRUBBED], [])
        self.assertClean('[assembly: InternalsVisibleTo("Foo.Tests, PublicKey=002400000480000094000000060200000024000052534131000400000100010027")]')
        self.assertClean('NAD1983StatePlaneAlabamaEastFIPS0101.GeographicInfo.Datum.Name = "D_North_American_1983";')
        self.assertClean('var id = $"{scope}/providers/Microsoft.Authorization/roleDefinitions/acdd72a7-3385-48ef-bd42-f606fba81ae7";')
        self.assertClean('//System.out.println("mine="+mine); x@out.println(y)')
        self.assertClean('cfg.Data = "product=1&name=66666&path=&password=&encoding=utf-8&client=&port=80"')
        self.assertEqual([m for m in sf.find_matches('// see https://github.com/owner/repo/blob/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0/pkg/x.go')
                          if m.category in sf.SCRUBBED], [])
        self.assertEqual([m for m in sf.find_matches('// see https://github.com/owner/repo/blob/' + tok(40) + '/pkg/x.go')
                          if m.category in sf.SCRUBBED], [])
        self.assertClean('dsn := "root:root@tcp(127.0.0.1:3306)/db"')
        self.assertEqual(sf.classify_line('dsn := "root:CxzhD624r27jh@tcp(9.135.1.2:3306)/db"'), "conn_string")

    def test_placeholder_connection_strings(self):
        self.assertClean('postgres://user:password@localhost:5432/db')
        self.assertClean('postgres://postgres:postgres@localhost/postgres')
        self.assertClean('amqp://guest:guest@localhost:5672/')
        self.assertClean('mysql://user:pass@localhost/db')
        self.assertClean('"root:@tcp(127.0.0.1:3306)/test"')
        self.assertClean('https://user:%s@host/')
        self.assertClean('Server=localhost;Database=test;Trusted_Connection=True;')
        self.assertClean('Server=.;Database=x;User Id=sa;Password=yourStrong(!)Password;')

    def test_hashes_uuids_blobs(self):
        for line in ('want := "' + SHA256 + '"', '"' + SHA1 + '"', MD5, UUID, 'id := "' + UUID + '"',
                     UUID.upper(), 'img := "data:image/png;base64,' + PNG_B64 + '"', 'png := "' + PNG_B64 + '"'):
            ms = [m for m in sf.find_matches(line) if m.category in sf.SCRUBBED]
            self.assertEqual(ms, [], msg=line)

    def test_public_keys_and_certs_not_flagged(self):
        for text in (PUB, CERT, '[assembly: InternalsVisibleTo("Foo.Tests, PublicKey=0024000004800000940000000602000000240000525341310004000001000100b5fc90e7027f67871e773a8fde8938c81dd402ba65b9201d60593e96c492651e889cc13f1415ebb53fac1131ae0bd333c5ee6021672d9718ea31a8aeb3af3a3fef9b2d1ab2ee0a4e1a5c1a1b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f7081")]',
                     'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC7vbqajDhA6Z8LqkR2Pn9k3Jx5mT1vQ8wZ0yX4dF6gH2jK9lM3nB7cV5xZ1aS4dF6gH8jK0lM2nB4cV6xZ8aS0dF2gH4jK6lM8nB0cV2xZ4aS6dF8gH0jK2lM4nB6cV8xZ0a user@host',
                     'var fileDescriptor_0 = []byte{0x0a, 0x0b, 0x0c}'):
            ms = [m for m in sf.find_matches(text) if m.category in sf.SCRUBBED]
            self.assertEqual(ms, [], msg=text[:60])

    def test_identifiers_not_high_entropy(self):
        self.assertClean('func TestReconcileDeploymentWithMultipleReplicaSets2(t *testing.T) {')
        self.assertClean('const MaxConcurrentReconcilesPerControllerDefaultValue1 = 1')
        self.assertClean('http_requests_total_by_status_code_and_method_v2 = 1')
        self.assertClean('// next step in docs/guardaccuracy/evaluation/phase2/results1234/summary.md')
        self.assertClean('applyset.kubernetes.io/id: applyset-v1-Jk7Qm2Lp9Xw4Rt1Vb5Nz6Hh0Fg3Cd2Ea8Yx-v1')
        self.assertClean('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/')  # alphabet table, no digits<3? has digits

    def test_private_and_doc_ips(self):
        for ip in ("10.0.0.1", "172.16.5.4", "172.31.255.255", "192.168.1.1", "127.0.0.1", "0.0.0.0", "169.254.169.254",
                   "224.0.0.1", "255.255.255.255", "192.0.2.1", "198.51.100.7", "203.0.113.9", "100.64.0.1", "198.18.0.1"):
            self.assertClean(f'addr := "{ip}:80"')
        self.assertClean('[assembly: AssemblyVersion("1.0.0.0")]')
        self.assertClean('[assembly: AssemblyFileVersion("4.5.2.0")]')
        self.assertClean('version = "1.2.3.4"')  # well-known dummy -> report only as ip_wellknown? no: version context
        self.assertClean('Net.IPv4(1, 2, 3, 4)')
        self.assertClean('v1.2.3.4')
        self.assertClean('"4.5.6.7.8"')
        self.assertClean('// see 12.6.4.12 and 7.3.4.2')
        self.assertClean('// 3.3.5.2: If Connection.SupportsMultiCredit is FALSE the client MUST remove it')
        self.assertClean('x := "3.3.5.2"')  # all octets <= 20 without network context: a version
        self.assertClean('Version: "3.14.1.42",')
        self.assertClean('// e.g. "4.1.0", "4.0.6", or "4.1.0.123"')
        self.assertEqual(sf.classify_line('{"Quake", mustCIDR("180.163.220.0/24")},'), "public_ip")
        self.assertEqual(sf.classify_line('Endpoints: []string{"https://8.6.4.21:2379"}'), "public_ip")
        self.assertClean('versions = new[] { "2.5.1.3", "2.5.1.4" };')
        self.assertEqual(sf.classify_line('ip := "3.3.5.2"'), "public_ip")
        self.assertEqual(sf.classify_line('x := "13.3.5.22:8080"'), "public_ip")
        self.assertEqual(sf.classify_line('addr := "12.6.4.12:443"'), "public_ip")

    def test_allowlisted_emails(self):
        for line in ('// Copyright 2019 Jane Roe <jane.roe@realcorp.com>', 'noreply@realcorp.com', 'git@github.com:foo/bar.git',
                     '"https://x-access-token:tok@github.com/org/repo.git"',
                     'user@example.com', 'test@test.com', 'foo@localhost', 'someone@example.org', 'admin@contoso.com',
                     '// Author: Jane Roe <jane.roe@realcorp.com>', 'Signed-off-by: J R <jr@realcorp.com>',
                     'github.com/foo/bar@v1.2.3', 'bar@v0.0.0-20200101000000-abcdef123456', '12345678+jane@users.noreply.github.com'):
            ms = [m for m in sf.find_matches(line) if m.category == "email"]
            self.assertEqual(ms, [], msg=line)

    def test_non_phones(self):
        self.assertClean('const x = 1234567890')
        self.assertClean('t := time.Date(2020, 1, 2, 3, 4, 5, 0, time.UTC)')


class Scrub(unittest.TestCase):
    def test_drop_private_key(self):
        text = "package x\n\nconst key = `" + PRIV + "`\n"
        out, st = sf.scrub(text, "x.go")
        self.assertIsNone(out)
        self.assertEqual(st["dropped"], "private_key")
        out2, st2 = sf.scrub(text, "x.go", {"drop_on_private_key": False, "drop_on_credential_hits": 3, "scrub": list(sf.SCRUBBED)})
        self.assertEqual(out2, "package x\n\nconst key = `<redacted-private-key>\n`\n")

    def test_drop_data_file(self):
        text = "var ips = []string{\n" + "".join(f'\t"{51 + i % 100}.{15 + i // 100}.{(i * 7) % 250}.{(i * 13) % 250 + 1}",\n' for i in range(150)) + "}\n"
        out, st = sf.scrub(text)
        self.assertIsNone(out)
        self.assertEqual(st["dropped"], "data_file")
        self.assertGreaterEqual(st["matches"]["public_ip"], 100)
        out2, st2 = sf.scrub(text, "", dict(sf.DEFAULT_POLICY, drop_on_scrub_matches=0))
        self.assertIsNotNone(out2)
        self.assertEqual(out2.count("203.0.113.1"), st2["matches"]["public_ip"])

    def test_drop_three_credentials(self):
        text = (f'a := "{AWS_KEY}"\n'
                'b := "postgres://app:Tr0ub4dor3@db.internal/app"\n'
                'c := "' + JWT + '"\n')
        out, st = sf.scrub(text)
        self.assertIsNone(out)
        self.assertEqual(st["dropped"], "credential_hits")

    def test_value_replacement_keeps_shape(self):
        text = (f'apiKey := "{GOOGLE_KEY}"\n'
                'dsn := "postgres://app:Tr0ub4dor3@db.internal:5432/app?sslmode=disable"\n'
                'mail := "bob.smith1984@gmail.com"\n'
                'ip := "51.15.236.17"\n'
                'ok := os.Getenv("API_KEY")\n')
        out, st = sf.scrub(text)
        self.assertEqual(out, ('apiKey := "<redacted-key>"\n'
                               'dsn := "postgres://app:<redacted-password>@db.internal:5432/app?sslmode=disable"\n'
                               'mail := "user@example.com"\n'
                               'ip := "203.0.113.1"\n'
                               'ok := os.Getenv("API_KEY")\n'))
        self.assertEqual(st["lines_scrubbed"], 4)
        self.assertEqual(st["dropped"], None)
        self.assertEqual(st["bytes_affected"], 39 + 10 + 23 + 12)

    def test_clean_file_untouched(self):
        text = 'package main\n\nimport "fmt"\n\nfunc main() { fmt.Printf("password=%s\\n", "***") }\n'
        out, st = sf.scrub(text)
        self.assertIs(out, text)
        self.assertEqual(st["matches"], {})
        data = text.encode()
        out_b, _ = sf.scrub_bytes(data)
        self.assertIs(out_b, data)

    def test_report_only_not_scrubbed(self):
        text = 'h := "' + SHA256 + '"\nd := "8.8.8.8"\nm := "user@example.com"\n'
        out, st = sf.scrub(text)
        self.assertEqual(out, text)

    def test_bytes_roundtrip_non_utf8(self):
        data = b'x := "51.15.236.17" // caf\xe9 \xff\n'
        out, st = sf.scrub_bytes(data)
        self.assertEqual(out, b'x := "203.0.113.1" // caf\xe9 \xff\n')

    def test_mask_never_leaks(self):
        text = f'k := "{AWS_KEY}"; e := "bob.smith1984@gmail.com"; password := "Tr0ub4dor3xyz"'
        for m in sf.find_matches(text):
            mk = sf.mask(m, text)
            self.assertNotIn(m.value(text), mk) if m.category != "private_key" else None
        line = sf.masked_line(text, sf.find_matches(text), 1)
        self.assertNotIn("Z7Q3K1XW9P2YB8DT", line)
        self.assertNotIn("smith1984", line)
        self.assertNotIn("ub4dor3", line)

    def test_is_test_path(self):
        for p in ("pkg/foo_test.go", "testdata/x.go", "internal/fixtures/a.go", "examples/demo/main.go",
                  "src/Foo.Tests/BarTests.cs", "test/Foo.cs", "Mocks/FooMock.cs", "src/FooTest.cs"):
            self.assertTrue(sf.is_test_path(p), p)
        for p in ("pkg/foo.go", "cmd/server/main.go", "src/Foo/Bar.cs", "src/Contest/Scoring.cs", "attestation.go"):
            self.assertFalse(sf.is_test_path(p), p)


if __name__ == "__main__":
    unittest.main()
