// CSI receiver: associates with the transmitter SoftAP, enables CSI capture,
// and streams every CSI sample over UART as a single line:
//
//   CSI <seq> <ts_us> <rssi> <noise> <ch> <bw> <mcs> <len> <base64(buf)>
//
// `buf` is the raw int8 IQ pair array from esp_wifi (imag, real, imag, real, ...).
// The host decodes base64 and reconstructs complex subcarriers from the pairs.

#include <string.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "rom/ets_sys.h"

static const char *TAG = "csi_rx";

static EventGroupHandle_t s_wifi_events;
#define WIFI_CONNECTED_BIT BIT0

static uint8_t s_ap_bssid[6];
static bool s_ap_bssid_known = false;

static const char b64_alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Encode a binary buffer to base64 directly into a caller-provided output buffer.
// out_buf must be at least 4 * ceil(len / 3) + 1 bytes.
static int b64_encode(const uint8_t *in, size_t len, char *out) {
    size_t i = 0, j = 0;
    while (i + 3 <= len) {
        uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8) | in[i + 2];
        out[j++] = b64_alphabet[(v >> 18) & 0x3F];
        out[j++] = b64_alphabet[(v >> 12) & 0x3F];
        out[j++] = b64_alphabet[(v >> 6) & 0x3F];
        out[j++] = b64_alphabet[v & 0x3F];
        i += 3;
    }
    if (i < len) {
        uint32_t v = (uint32_t)in[i] << 16;
        if (i + 1 < len) v |= (uint32_t)in[i + 1] << 8;
        out[j++] = b64_alphabet[(v >> 18) & 0x3F];
        out[j++] = b64_alphabet[(v >> 12) & 0x3F];
        out[j++] = (i + 1 < len) ? b64_alphabet[(v >> 6) & 0x3F] : '=';
        out[j++] = '=';
    }
    out[j] = '\0';
    return j;
}

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "disconnected, retrying");
        s_ap_bssid_known = false;
        xEventGroupClearBits(s_wifi_events, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
        wifi_event_sta_connected_t *e = data;
        memcpy(s_ap_bssid, e->bssid, 6);
        s_ap_bssid_known = true;
        ESP_LOGI(TAG, "connected to ssid=%s ch=%d", e->ssid, e->channel);
        xEventGroupSetBits(s_wifi_events, WIFI_CONNECTED_BIT);
    }
}

static void csi_callback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len <= 0) return;

#if CONFIG_CSI_RX_FILTER_BY_BSSID
    if (!s_ap_bssid_known) return;
    if (memcmp(info->mac, s_ap_bssid, 6) != 0) return;
#endif

    static uint32_t seq = 0;
    // Worst-case base64 of 384 bytes (HT40 LLTF+HT-LTF) is 512 chars + null.
    static char b64[1024];
    int n = b64_encode((const uint8_t *)info->buf, info->len, b64);
    (void)n;

    int64_t ts = esp_timer_get_time();
    wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;

    // Single printf keeps the whole line atomic on the UART.
    printf("CSI %lu %lld %d %d %d %d %d %d %s\n",
           (unsigned long)seq++,
           (long long)ts,
           rx->rssi,
           rx->noise_floor,
           rx->channel,
           rx->cwb,
           rx->mcs,
           info->len,
           b64);
}

static void wifi_init_sta(void) {
    s_wifi_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = CONFIG_CSI_RX_AP_SSID,
            .password = CONFIG_CSI_RX_AP_PASSWORD,
            .threshold.authmode = WIFI_AUTH_OPEN,
        },
    };
    if (strlen(CONFIG_CSI_RX_AP_PASSWORD) > 0) {
        wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
}

static void enable_csi(void) {
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

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init_sta();
    xEventGroupWaitBits(s_wifi_events, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    enable_csi();
    ESP_LOGI(TAG, "CSI capture started; streaming over UART");
}
