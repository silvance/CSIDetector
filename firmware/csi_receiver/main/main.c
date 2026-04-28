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
static volatile uint32_t s_csi_total_calls = 0;
static volatile uint32_t s_csi_emitted = 0;
static volatile uint32_t s_promisc_count = 0;

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    c = tolower((unsigned char)c);
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    return -1;
}

static bool parse_filter_mac(const char *s, uint8_t out[6]) {
    if (!s || strlen(s) != 12) return false;
    for (int i = 0; i < 6; i++) {
        int hi = hex_nibble(s[i * 2]);
        int lo = hex_nibble(s[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

static void csi_callback(void *ctx, wifi_csi_info_t *info) {
    s_csi_total_calls++;
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

static void promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type) {
    s_promisc_count++;
}

static void wifi_init(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    // WIFI_MODE_NULL avoids spinning up the STA TX path we don't need
    // (we never associate). Matches esp-csi/csi_recv.
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_NULL));
    ESP_ERROR_CHECK(esp_wifi_start());
    // Promiscuous must come before set_channel in IDF v5.x.
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    // Without an explicit filter, only mgmt frames are captured by default
    // and CSI never fires for the data frames we care about. Capture all.
    wifi_promiscuous_filter_t filt = { .filter_mask = WIFI_PROMIS_FILTER_MASK_ALL };
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&filt));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_rx_cb(&promiscuous_cb));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_CSI_RX_CHANNEL, WIFI_SECOND_CHAN_NONE));
}

static void espnow_init(void) {
    ESP_ERROR_CHECK(esp_now_init());
    // No peers required for receive — the broadcast destination is implicit.
}

static void enable_csi(void) {
    // Defaults from esp-csi examples/get-started/csi_recv (line ~195-203).
    wifi_csi_config_t cfg = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = 0,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&csi_callback, NULL));
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

    s_filter_active = parse_filter_mac(CONFIG_CSI_RX_FILTER_TX_MAC, s_filter_mac);
    if (s_filter_active) {
        ESP_LOGI(TAG, "filtering on TX MAC %02x:%02x:%02x:%02x:%02x:%02x",
                 s_filter_mac[0], s_filter_mac[1], s_filter_mac[2],
                 s_filter_mac[3], s_filter_mac[4], s_filter_mac[5]);
    }

    ESP_LOGI(TAG, "BISECT: pre-wifi_init");
    vTaskDelay(pdMS_TO_TICKS(200));
    wifi_init();
    ESP_LOGI(TAG, "BISECT: post-wifi_init");
    vTaskDelay(pdMS_TO_TICKS(200));

    espnow_init();
    ESP_LOGI(TAG, "BISECT: post-espnow_init");
    vTaskDelay(pdMS_TO_TICKS(200));

    emit_header();
    enable_csi();
    ESP_LOGI(TAG, "BISECT: post-enable_csi (CSI capture started on channel %d)",
             CONFIG_CSI_RX_CHANNEL);

    while (1) {
        ESP_LOGI(TAG, "BISECT: alive promisc=%lu csi_calls=%lu emitted=%lu",
                 (unsigned long)s_promisc_count,
                 (unsigned long)s_csi_total_calls,
                 (unsigned long)s_csi_emitted);
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
