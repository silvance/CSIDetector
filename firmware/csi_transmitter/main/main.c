// CSI transmitter: broadcasts ESP-NOW frames at a fixed rate on a fixed
// channel. Each broadcast triggers one CSI sample on every receiver tuned
// to the same channel — no association, no DHCP. Mirrors the convention
// in espressif/esp-csi examples/get-started/csi_send.

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_mac.h"
#include "nvs_flash.h"

static const char *TAG = "csi_tx";

static const uint8_t BROADCAST_MAC[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

static SemaphoreHandle_t s_send_complete;

static void wifi_init(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    // Force HT20 / 11n. ESP-NOW broadcasts default to legacy 11b without
    // HT-LTF, which means the receiver's CSI engine never fires for our
    // frames. Selecting 11n + HT20 makes every broadcast carry HT-LTF so
    // CSI samples are produced. Matches esp-csi/csi_send.
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA,
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));

    // Pin to a fixed channel so receivers can stay tuned.
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_CSI_TX_CHANNEL, WIFI_SECOND_CHAN_NONE));
}

static void on_send_done(const uint8_t *mac, esp_now_send_status_t status) {
    (void)mac;
    (void)status;
    xSemaphoreGive(s_send_complete);
}

static void espnow_init(void) {
    s_send_complete = xSemaphoreCreateBinary();
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(on_send_done));
    esp_now_peer_info_t peer = {
        .channel = CONFIG_CSI_TX_CHANNEL,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
    };
    memcpy(peer.peer_addr, BROADCAST_MAC, 6);
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
}

static void broadcast_task(void *arg) {
    const TickType_t period = pdMS_TO_TICKS(1000 / CONFIG_CSI_TX_RATE_HZ);
    uint32_t seq = 0;
    uint32_t sent = 0;
    uint32_t dropped = 0;
    uint8_t payload[16];
    TickType_t next = xTaskGetTickCount();
    TickType_t last_report = next;
    while (1) {
        // Gate on previous send: the WiFi TX queue is shallow (2 FG buffers
        // by default), so if we don't wait for completion we'll saturate it
        // under any RF contention and esp_now_send starts returning NO_MEM.
        xSemaphoreTake(s_send_complete, pdMS_TO_TICKS(100));

        memcpy(payload, &seq, sizeof(seq));
        memset(payload + sizeof(seq), 0xA5, sizeof(payload) - sizeof(seq));
        esp_err_t err = esp_now_send(BROADCAST_MAC, payload, sizeof(payload));
        if (err == ESP_OK) {
            sent++;
        } else {
            dropped++;
            if (err != ESP_ERR_ESPNOW_NO_MEM) {
                ESP_LOGW(TAG, "esp_now_send: %s", esp_err_to_name(err));
            }
        }
        seq++;

        TickType_t now = xTaskGetTickCount();
        if ((now - last_report) >= pdMS_TO_TICKS(2000)) {
            ESP_LOGI(TAG, "heartbeat: sent=%lu dropped=%lu",
                     (unsigned long)sent, (unsigned long)dropped);
            last_report = now;
        }

        vTaskDelayUntil(&next, period);
    }
}

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init();
    espnow_init();

    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "TX up: mac=%02x:%02x:%02x:%02x:%02x:%02x ch=%d rate=%dHz",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
             CONFIG_CSI_TX_CHANNEL, CONFIG_CSI_TX_RATE_HZ);

    xTaskCreate(broadcast_task, "broadcast", 4096, NULL, 5, NULL);
}
