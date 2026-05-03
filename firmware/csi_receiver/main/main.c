// CSI receiver: tunes to the transmitter's channel and prints CSI samples
// over UART in the esp-csi line format so existing host tooling
// (esp-csi/examples/get-started/tools/csi_data_read_parse.py) works
// untouched. Optionally also forwards each sample as a binary UDP packet
// to a host on a configured WiFi hotspot, so multiple receivers can
// stream into one PC without USB tethers (multi-RX heatmap setup).
//
// UART output format (single header at boot, then one row per CSI sample):
//
//   CSI_DATA,seq,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,
//            not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,
//            ampdu_cnt,channel,secondary_channel,local_timestamp,ant,
//            sig_len,rx_state,len,first_word,data
//
// `data` is a JSON int8 array of (Im, Re) pairs.
//
// UDP output format (when CSI_RX_WIFI_SSID is set): see csi_udp_header_t
// below. Little-endian, packed, header followed by `len` bytes of int8
// IQ data. Host parses by `version`.

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <ctype.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

static const char *TAG = "csi_rx";

// Up to 4 TX MACs in the filter — enough for any reasonable multi-TX
// localization setup. Comma- or semicolon-separated in Kconfig.
#define MAX_FILTER_MACS 4
static uint8_t s_filter_macs[MAX_FILTER_MACS][6];
static int s_filter_count = 0;

// UDP forwarding state. s_udp_sock is < 0 until the STA gets an IP and
// the socket is created. UART output continues regardless.
static int s_udp_sock = -1;
static struct sockaddr_in s_udp_dest;
static uint8_t s_rx_mac[6];

// Signaled by IP_EVENT_STA_GOT_IP. wifi_init() blocks on this so the
// boot only proceeds to enable_csi() once UDP forwarding is ready —
// otherwise CSI rows would print to UART for several seconds before
// any packets could leave the device.
static EventGroupHandle_t s_wifi_events;
#define WIFI_EVT_GOT_IP BIT0

// Wire format of the UDP packet header. Little-endian, packed; total
// 34 bytes, followed by `len` bytes of int8 IQ samples.
typedef struct __attribute__((packed)) csi_udp_header {
    uint8_t  version;       // 1
    uint8_t  reserved;
    uint8_t  rx_mac[6];     // factory MAC of this RX
    uint8_t  tx_mac[6];     // source MAC of the captured frame (TX)
    uint32_t seq;
    int64_t  ts_us;
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  channel;
    uint8_t  sig_mode;
    uint8_t  mcs;
    uint8_t  bandwidth;
    uint16_t len;
} csi_udp_header_t;
_Static_assert(sizeof(csi_udp_header_t) == 34, "csi_udp_header_t must be 34 bytes");

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    c = tolower((unsigned char)c);
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    return -1;
}

// Accept colon/hyphen separators inside a MAC, comma/semicolon between
// MACs. The TX firmware logs its MAC as aa:bb:cc:dd:ee:ff, so users
// paste that form; rejecting it silently disables the filter.
static bool parse_one_mac(const char *s, int len, uint8_t out[6]) {
    int nibbles = 0;
    int acc = 0;
    for (int i = 0; i < len; i++) {
        char c = s[i];
        if (c == ':' || c == '-' || c == ' ' || c == '\t') continue;
        int n = hex_nibble(c);
        if (n < 0) return false;
        acc = (acc << 4) | n;
        if ((nibbles & 1) == 1) {
            if (nibbles / 2 >= 6) return false;
            out[nibbles / 2] = (uint8_t)acc;
            acc = 0;
        }
        nibbles++;
    }
    return nibbles == 12;
}

// Parses the Kconfig string into s_filter_macs / s_filter_count. Returns
// number of valid MACs parsed; returns -1 on any unparseable token so
// the caller can warn loudly (silent partial filters are the trap that
// motivated the original parser cleanup).
static int parse_filter_macs(const char *s) {
    s_filter_count = 0;
    if (!s) return 0;
    const char *p = s;
    while (*p && s_filter_count < MAX_FILTER_MACS) {
        // Skip leading separators / whitespace.
        while (*p && (*p == ',' || *p == ';' || *p == ' ' || *p == '\t')) p++;
        if (!*p) break;
        const char *end = p;
        while (*end && *end != ',' && *end != ';') end++;
        if (!parse_one_mac(p, end - p, s_filter_macs[s_filter_count])) {
            return -1;
        }
        s_filter_count++;
        p = end;
    }
    return s_filter_count;
}

static bool mac_in_filter(const uint8_t mac[6]) {
    for (int i = 0; i < s_filter_count; i++) {
        if (memcmp(mac, s_filter_macs[i], 6) == 0) return true;
    }
    return false;
}

static void udp_send_sample(uint32_t seq, int64_t ts, const wifi_csi_info_t *info) {
    if (s_udp_sock < 0) return;
    // Stack-allocated packet: header + IQ. info->len is bounded by the
    // CSI engine to a few hundred bytes, well under MTU.
    uint8_t pkt[sizeof(csi_udp_header_t) + 384];
    if (info->len > (int)(sizeof(pkt) - sizeof(csi_udp_header_t))) return;

    csi_udp_header_t *h = (csi_udp_header_t *)pkt;
    h->version = 1;
    h->reserved = 0;
    memcpy(h->rx_mac, s_rx_mac, 6);
    memcpy(h->tx_mac, info->mac, 6);
    h->seq = seq;
    h->ts_us = ts;
    h->rssi = info->rx_ctrl.rssi;
    h->noise_floor = info->rx_ctrl.noise_floor;
    h->channel = info->rx_ctrl.channel;
    h->sig_mode = info->rx_ctrl.sig_mode;
    h->mcs = info->rx_ctrl.mcs;
    h->bandwidth = info->rx_ctrl.cwb;
    h->len = (uint16_t)info->len;
    memcpy(pkt + sizeof(csi_udp_header_t), info->buf, info->len);

    // Non-blocking; drops on full TX queue rather than stalling capture.
    sendto(s_udp_sock, pkt, sizeof(csi_udp_header_t) + info->len, MSG_DONTWAIT,
           (struct sockaddr *)&s_udp_dest, sizeof(s_udp_dest));
}

static void csi_callback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len <= 0) return;
    if (s_filter_count > 0 && !mac_in_filter(info->mac)) return;

    static uint32_t seq = 0;
    uint32_t this_seq = seq++;
    wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;
    int64_t ts = esp_timer_get_time();
    int8_t *buf = (int8_t *)info->buf;

    udp_send_sample(this_seq, ts, info);

    // Header columns match esp-csi conventions so existing tooling can
    // parse this format unchanged. The printf+putchar sequence isn't
    // atomic — concurrent log lines can interleave — but stdout is line-
    // buffered and the host parser tolerates the rare garbled row.
    printf("CSI_DATA,%lu,"                 // seq
           "%02x:%02x:%02x:%02x:%02x:%02x," // mac
           "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d," // rssi,rate,sig_mode,mcs,bw,smoothing,not_sounding,aggregation,stbc,fec_coding
           "%d,%d,%d,"                      // sgi,noise_floor,ampdu_cnt
           "%d,%d,%lld,"                    // channel,secondary_channel,local_timestamp
           "%d,%d,%d,%d,%d,",               // ant,sig_len,rx_state,len,first_word
           (unsigned long)this_seq,
           info->mac[0], info->mac[1], info->mac[2],
           info->mac[3], info->mac[4], info->mac[5],
           rx->rssi, rx->rate, rx->sig_mode, rx->mcs, rx->cwb,
           rx->smoothing, rx->not_sounding, rx->aggregation, rx->stbc, rx->fec_coding,
           rx->sgi, rx->noise_floor, rx->ampdu_cnt,
           rx->channel, rx->secondary_channel, (long long)ts,
           rx->ant, rx->sig_len, rx->rx_state, info->len, info->first_word_invalid ? 1 : 0);

    putchar('"');
    putchar('[');
    for (int i = 0; i < info->len; i++) {
        if (i > 0) putchar(',');
        printf("%d", buf[i]);
    }
    putchar(']');
    putchar('"');
    putchar('\n');
}

static void open_udp_socket(esp_ip4_addr_t ip) {
    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_udp_sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno %d", errno);
        return;
    }
    s_udp_dest.sin_family = AF_INET;
    s_udp_dest.sin_port = htons(CONFIG_CSI_RX_HOST_PORT);
    if (inet_pton(AF_INET, CONFIG_CSI_RX_HOST_IP, &s_udp_dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "bad CSI_RX_HOST_IP=%s", CONFIG_CSI_RX_HOST_IP);
        close(s_udp_sock);
        s_udp_sock = -1;
        return;
    }
    ESP_LOGI(TAG, "UDP forwarding to %s:%d (this RX " IPSTR ")",
             CONFIG_CSI_RX_HOST_IP, CONFIG_CSI_RX_HOST_PORT, IP2STR(&ip));
}

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data) {
    // STA_START is intentionally NOT handled here — we call
    // esp_wifi_connect() explicitly from wifi_init() after promisc
    // setup so the order is deterministic. Auto-connecting from the
    // STA_START handler races the main thread's promiscuous setup and
    // the connect silently fails on some IDF versions.
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
        ESP_LOGI(TAG, "STA associated to AP");
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *e = (wifi_event_sta_disconnected_t *)data;
        ESP_LOGW(TAG, "STA disconnected (reason=%d), reconnecting", e ? e->reason : 0);
        s_udp_sock = -1;
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)data;
        open_udp_socket(event->ip_info.ip);
        if (s_wifi_events) xEventGroupSetBits(s_wifi_events, WIFI_EVT_GOT_IP);
    }
}

static void wifi_init(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));

    bool wifi_enabled = (CONFIG_CSI_RX_WIFI_SSID[0] != '\0');
    if (wifi_enabled) {
        s_wifi_events = xEventGroupCreate();
        ESP_ERROR_CHECK(esp_event_handler_instance_register(
            WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
        ESP_ERROR_CHECK(esp_event_handler_instance_register(
            IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));
        wifi_config_t sta_cfg = {0};
        strlcpy((char *)sta_cfg.sta.ssid, CONFIG_CSI_RX_WIFI_SSID, sizeof(sta_cfg.sta.ssid));
        strlcpy((char *)sta_cfg.sta.password, CONFIG_CSI_RX_WIFI_PASS, sizeof(sta_cfg.sta.password));
        // Fast-scan on the configured channel so association doesn't
        // sweep all 13 channels.
        sta_cfg.sta.channel = CONFIG_CSI_RX_CHANNEL;
        sta_cfg.sta.scan_method = WIFI_FAST_SCAN;
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_cfg));
    }

    ESP_ERROR_CHECK(esp_wifi_start());

    // Set protocol + bandwidth BEFORE connecting. Calling these on an
    // already-associated STA forces a renegotiation and the AP drops
    // us (reason 8, LEAVING). Pre-connect they're harmless.
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA,
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N | WIFI_PROTOCOL_LR));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40));

    if (wifi_enabled) {
        // Associated path: the AP's channel becomes the radio's
        // channel automatically, and CSI fires for ESP-NOW broadcasts
        // (destination ff:ff:ff:ff:ff:ff, accepted by every STA at L2)
        // so promiscuous mode is unnecessary here. Enabling promisc
        // after association also disconnects the STA on IDF 5.x
        // (reason 8, LEAVING) — see commit history.
        esp_err_t err = esp_wifi_connect();
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "esp_wifi_connect failed: %s", esp_err_to_name(err));
        }
        EventBits_t bits = xEventGroupWaitBits(s_wifi_events, WIFI_EVT_GOT_IP,
                                               pdFALSE, pdTRUE,
                                               pdMS_TO_TICKS(10000));
        if (!(bits & WIFI_EVT_GOT_IP)) {
            ESP_LOGW(TAG, "STA didn't get IP within 10s — continuing in "
                          "capture-only mode (no UDP forwarding)");
        }
    } else {
        // Standalone mode (UART-only, no AP): need promiscuous to
        // capture anything, and the channel has to be pinned manually.
        ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
        wifi_promiscuous_filter_t filt = { .filter_mask = WIFI_PROMIS_FILTER_MASK_ALL };
        ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&filt));
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_CSI_RX_CHANNEL, WIFI_SECOND_CHAN_NONE));
    }

    // Cache our MAC for stamping outgoing UDP packets (the host uses it
    // to demux per-RX streams).
    ESP_ERROR_CHECK(esp_wifi_get_mac(WIFI_IF_STA, s_rx_mac));
}

static void espnow_init(void) {
    ESP_ERROR_CHECK(esp_now_init());
    // No peers required for receive — the broadcast destination is implicit.
}

static void enable_csi(void) {
    wifi_csi_config_t cfg = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        // false: accept CSI from HT20 frames even though we're in HT40.
        // The TX is pinned to HT20 (so its ESP-NOW broadcasts carry HT-LTF),
        // and channel_filter_en=true silently drops bandwidth-mismatched
        // frames from the CSI path — promisc still sees them.
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
    };
    // esp-csi/csi_recv order: rx_cb -> config -> enable. Some IDF versions
    // silently drop set_csi_config if it lands before set_csi_rx_cb.
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

static void emit_header(void) {
    printf("type,seq,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,"
           "not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,"
           "ampdu_cnt,channel,secondary_channel,local_timestamp,ant,"
           "sig_len,rx_state,len,first_word,data\n");
}

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    const char *cfg_mac = CONFIG_CSI_RX_FILTER_TX_MAC;
    int parsed = parse_filter_macs(cfg_mac);
    if (parsed > 0) {
        for (int i = 0; i < s_filter_count; i++) {
            ESP_LOGI(TAG, "filtering on TX MAC[%d] %02x:%02x:%02x:%02x:%02x:%02x",
                     i,
                     s_filter_macs[i][0], s_filter_macs[i][1], s_filter_macs[i][2],
                     s_filter_macs[i][3], s_filter_macs[i][4], s_filter_macs[i][5]);
        }
    } else if (parsed < 0) {
        // Non-empty but unparseable somewhere — surface this loudly so the
        // user doesn't think they're filtering when they aren't.
        ESP_LOGE(TAG, "CSI_RX_FILTER_TX_MAC=\"%s\" is not a valid MAC list; "
                      "filter DISABLED. Use aa:bb:cc:dd:ee:ff or comma-separated "
                      "for multiple TXs.",
                 cfg_mac);
    } else {
        ESP_LOGW(TAG, "no TX MAC filter set — emitting CSI for every frame "
                      "the radio decodes (set CSI_RX_FILTER_TX_MAC to clean up)");
    }

    wifi_init();
    espnow_init();
    emit_header();
    enable_csi();
    ESP_LOGI(TAG, "CSI capture started on channel %d", CONFIG_CSI_RX_CHANNEL);
}
