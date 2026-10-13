// Board settings written by the setup app at flash time.
//
// The firmware is built once per chip and shipped prebuilt; what differs
// per board (Wi-Fi channel, transmitter MACs, hotspot SSID/password,
// label…) lives in a 4 KB "csi_cfg" partition the setup app writes
// alongside the app image. Layout (little-endian):
//
//   0   "CSIS"          magic
//   4   u8  version     1
//   5   u8  reserved    0
//   6   u16 length      payload bytes
//   8   u32 crc32       of the payload (zlib / IEEE 802.3)
//   12  payload         UTF-8 "key=value\n" lines; unknown keys ignored
//
// Anything else (an erased partition reads 0xFF) means "not set by the
// setup app": the Kconfig defaults stay in force, so `idf.py flash` from
// a developer's tree still works as before.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSI_SETTINGS_LABEL     "csi_cfg"
#define CSI_SETTINGS_SUBTYPE   0x40
#define CSI_SETTINGS_MAX_SIZE  4096
#define CSI_SETTINGS_HEADER    12

typedef struct {
    bool from_setup;        // true when the values came from the settings block
    char role[4];           // "tx" / "rx" (informational)
    char label[64];         // the board's name / location, e.g. "Kitchen RX"
    int  channel;           // 1..13
    int  rate_hz;           // transmitter broadcast rate, 1..500
    char tx_macs[128];      // receiver MAC filter, "aa:bb:..., ..." ("" = any)
    char ssid[33];          // receiver hotspot SSID ("" = USB/UART only)
    char password[65];
    char host_ip[16];
    int  host_port;         // 1..65535
} csi_settings_t;

enum {
    CSI_SETTINGS_OK = 0,
    CSI_SETTINGS_ERR_EMPTY = -1,     // no settings block (erased / never written)
    CSI_SETTINGS_ERR_VERSION = -2,   // written by a newer setup app
    CSI_SETTINGS_ERR_FORMAT = -3,    // length out of range, or a line without '='
    CSI_SETTINGS_ERR_CRC = -4,       // damaged
    CSI_SETTINGS_ERR_VALUE = -5,     // a value too long or out of range
};

// zlib-compatible CRC-32 (what Python's zlib.crc32 computes).
uint32_t csi_settings_crc32(const uint8_t *data, size_t len);

// Parse a settings block over the defaults already in *s. On any error *s
// is left exactly as it was. Pure C: also compiled by the host test suite.
int csi_settings_parse(const uint8_t *blob, size_t len, csi_settings_t *s);

const char *csi_settings_strerror(int err);

#ifdef ESP_PLATFORM
#include "esp_err.h"
// Read the csi_cfg partition into *s (which holds the Kconfig defaults)
// and log where the settings came from. Never fails the boot: a missing
// or bad block keeps the defaults.
void csi_settings_load(csi_settings_t *s);
#endif

#ifdef __cplusplus
}
#endif
