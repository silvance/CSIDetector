// CSI receiver: tunes to the transmitter's channel and prints CSI samples
// over UART in the esp-csi line format so existing host tooling
// (esp-csi/examples/get-started/tools/csi_data_read_parse.py) works
// untouched. ESP-NOW is initialized so the radio decodes the broadcast
// frames and fires the CSI callback; we do not consume the payloads.
//
// Output line format (single header at boot, then one row per CSI sample):
//
//   CSI_DATA,seq,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,
//            not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,
//            ampdu_cnt,channel,secondary_channel,local_timestamp,ant,
//            sig_len,rx_state,len,first_word,data
//
// `data` is a JSON int8 array of (Im, Re) pairs.

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <ctype.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "nvs_flash.h"

static const char *TAG = "csi_rx";

static uint8_t s_filter_mac[6];
static bool s_filter_active = false;

// Diagnostic counters. Help separate "TX frames don't reach the radio"
// (promisc_match stays 0) from "they reach the radio but no CSI fires"
// (promisc_match grows, csi_emitted stays 0).
static volatile uint32_t s_promisc_total = 0;
static volatile uint32_t s_promisc_match = 0;
static volatile uint32_t s_csi_total = 0;
static volatile uint32_t s_csi_emitted = 0;

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    c = tolower((unsigned char)c);
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    return -1;
}

// Accept colon/hyphen separators too: the TX firmware logs its MAC as
// aa:bb:cc:dd:ee:ff, and rejecting that form silently disables the filter.
static bool parse_filter_mac(const char *s, uint8_t out[6]) {
    if (!s) return false;
    int nibbles = 0;
    int acc = 0;
    for (const char *p = s; *p; p++) {
        if (*p == ':' || *p == '-' || *p == ' ') continue;
        int n = hex_nibble(*p);
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

static void promisc_callback(void *buf, wifi_promiscuous_pkt_type_t type) {
    (void)type;
    s_promisc_total++;
    if (!s_filter_active) return;
    // 802.11 A2 (transmitter address) lives at byte 10 of every
    // non-control frame — same offset for management/data, and that's
    // what wifi_csi_info_t.mac is supposed to mirror.
    const wifi_promiscuous_pkt_t *pkt = (const wifi_promiscuous_pkt_t *)buf;
    if (memcmp(pkt->payload + 10, s_filter_mac, 6) == 0) {
        s_promisc_match++;
    }
}

static void diag_timer_cb(void *arg) {
    (void)arg;
    ESP_LOGI(TAG, "diag: promisc_total=%lu promisc_match=%lu csi_total=%lu csi_emitted=%lu",
             (unsigned long)s_promisc_total,
             (unsigned long)s_promisc_match,
             (unsigned long)s_csi_total,
             (unsigned long)s_csi_emitted);
}

static void csi_callback(void *ctx, wifi_csi_info_t *info) {
    s_csi_total++;
    if (!info || !info->buf || info->len <= 0) return;
    if (s_filter_active && memcmp(info->mac, s_filter_mac, 6) != 0) return;
    s_csi_emitted++;

    static uint32_t seq = 0;
    wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;
    int64_t ts = esp_timer_get_time();
    int8_t *buf = (int8_t *)info->buf;

    // Header columns match esp-csi conventions; emitting them up front lets
    // a single fwrite per row stay atomic.
    printf("CSI_DATA,%lu,"                 // seq
           "%02x:%02x:%02x:%02x:%02x:%02x," // mac
           "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d," // rssi,rate,sig_mode,mcs,bw,smoothing,not_sounding,aggregation,stbc,fec_coding
           "%d,%d,%d,"                      // sgi,noise_floor,ampdu_cnt
           "%d,%d,%lld,"                    // channel,secondary_channel,local_timestamp
           "%d,%d,%d,%d,%d,",               // ant,sig_len,rx_state,len,first_word
           (unsigned long)seq++,
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

static void wifi_init(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    // 11B|11G|11N|LR + HT40, permissive promisc filter so CSI fires for
    // data frames too. HT20 was tried (to match the TX's HT20 broadcasts)
    // and wedged the chip — no frames decoded at all. Stay HT40.
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA,
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N | WIFI_PROTOCOL_LR));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40));
    wifi_promiscuous_filter_t filt = { .filter_mask = WIFI_PROMIS_FILTER_MASK_ALL };
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&filt));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_rx_cb(&promisc_callback));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_CSI_RX_CHANNEL, WIFI_SECOND_CHAN_NONE));
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
    s_filter_active = parse_filter_mac(cfg_mac, s_filter_mac);
    if (s_filter_active) {
        ESP_LOGI(TAG, "filtering on TX MAC %02x:%02x:%02x:%02x:%02x:%02x",
                 s_filter_mac[0], s_filter_mac[1], s_filter_mac[2],
                 s_filter_mac[3], s_filter_mac[4], s_filter_mac[5]);
    } else if (cfg_mac && cfg_mac[0]) {
        // Non-empty but unparseable: surface this loudly so the user
        // doesn't think they're filtering when they aren't.
        ESP_LOGE(TAG, "CSI_RX_FILTER_TX_MAC=\"%s\" is not a valid MAC; "
                      "filter DISABLED. Use aa:bb:cc:dd:ee:ff or aabbccddeeff.",
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

    const esp_timer_create_args_t diag_args = {
        .callback = &diag_timer_cb,
        .name = "csi_diag",
    };
    esp_timer_handle_t diag_timer;
    ESP_ERROR_CHECK(esp_timer_create(&diag_args, &diag_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(diag_timer, 2 * 1000 * 1000));
}
