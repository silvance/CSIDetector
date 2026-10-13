// Settings-block parser. No ESP-IDF dependencies: the host test suite
// compiles this file with the system C compiler and checks it against the
// setup app's encoder (host/tests/test_board_settings.py).

#include "csi_settings.h"

#include <stdlib.h>
#include <string.h>

uint32_t csi_settings_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    while (len--) {
        crc ^= *data++;
        for (int k = 0; k < 8; k++) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

static int set_str(char *dst, size_t cap, const char *v, size_t n) {
    if (n >= cap || memchr(v, '\0', n)) return CSI_SETTINGS_ERR_VALUE;
    memcpy(dst, v, n);
    dst[n] = '\0';
    return CSI_SETTINGS_OK;
}

static int set_int(int *dst, const char *v, size_t n, long lo, long hi) {
    char tmp[12];
    if (n == 0 || n >= sizeof(tmp)) return CSI_SETTINGS_ERR_VALUE;
    memcpy(tmp, v, n);
    tmp[n] = '\0';
    char *end = NULL;
    long x = strtol(tmp, &end, 10);
    if (*end != '\0' || x < lo || x > hi) return CSI_SETTINGS_ERR_VALUE;
    *dst = (int)x;
    return CSI_SETTINGS_OK;
}

static int set_key(csi_settings_t *s, const char *k, size_t kn, const char *v, size_t vn) {
#define KEY(name) (kn == sizeof(name) - 1 && memcmp(k, name, kn) == 0)
    if (KEY("role"))      return set_str(s->role, sizeof(s->role), v, vn);
    if (KEY("label"))     return set_str(s->label, sizeof(s->label), v, vn);
    if (KEY("channel"))   return set_int(&s->channel, v, vn, 1, 13);
    if (KEY("rate_hz"))   return set_int(&s->rate_hz, v, vn, 1, 500);
    if (KEY("tx_macs"))   return set_str(s->tx_macs, sizeof(s->tx_macs), v, vn);
    if (KEY("ssid"))      return set_str(s->ssid, sizeof(s->ssid), v, vn);
    if (KEY("password"))  return set_str(s->password, sizeof(s->password), v, vn);
    if (KEY("host_ip"))   return set_str(s->host_ip, sizeof(s->host_ip), v, vn);
    if (KEY("host_port")) return set_int(&s->host_port, v, vn, 1, 65535);
#undef KEY
    return CSI_SETTINGS_OK;     // a key from a newer setup app: ignore
}

int csi_settings_parse(const uint8_t *blob, size_t len, csi_settings_t *s) {
    if (!blob || len < CSI_SETTINGS_HEADER || memcmp(blob, "CSIS", 4) != 0) {
        return CSI_SETTINGS_ERR_EMPTY;
    }
    if (blob[4] != 1) return CSI_SETTINGS_ERR_VERSION;
    size_t n = (size_t)blob[6] | ((size_t)blob[7] << 8);
    if (n > len - CSI_SETTINGS_HEADER) return CSI_SETTINGS_ERR_FORMAT;
    uint32_t crc = (uint32_t)blob[8] | ((uint32_t)blob[9] << 8)
                 | ((uint32_t)blob[10] << 16) | ((uint32_t)blob[11] << 24);
    const uint8_t *payload = blob + CSI_SETTINGS_HEADER;
    if (csi_settings_crc32(payload, n) != crc) return CSI_SETTINGS_ERR_CRC;

    csi_settings_t t = *s;      // all or nothing
    const char *p = (const char *)payload;
    const char *end = p + n;
    while (p < end) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        if (!nl) nl = end;
        if (nl > p) {
            const char *eq = memchr(p, '=', (size_t)(nl - p));
            if (!eq) return CSI_SETTINGS_ERR_FORMAT;
            int rc = set_key(&t, p, (size_t)(eq - p), eq + 1, (size_t)(nl - eq - 1));
            if (rc != CSI_SETTINGS_OK) return rc;
        }
        if (nl == end) break;
        p = nl + 1;
    }
    t.from_setup = true;
    *s = t;
    return CSI_SETTINGS_OK;
}

const char *csi_settings_strerror(int err) {
    switch (err) {
    case CSI_SETTINGS_OK:          return "ok";
    case CSI_SETTINGS_ERR_EMPTY:   return "no settings written";
    case CSI_SETTINGS_ERR_VERSION: return "written by a newer setup app";
    case CSI_SETTINGS_ERR_FORMAT:  return "malformed";
    case CSI_SETTINGS_ERR_CRC:     return "damaged (checksum mismatch)";
    case CSI_SETTINGS_ERR_VALUE:   return "a value is out of range";
    default:                       return "unknown error";
    }
}
