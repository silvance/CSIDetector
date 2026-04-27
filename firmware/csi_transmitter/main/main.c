// CSI transmitter: broadcasts ESP-NOW frames at a fixed rate on a fixed
// channel. Each broadcast triggers one CSI sample on every receiver tuned
// to the same channel — no association, no DHCP. Mirrors the convention
// in espressif/esp-csi examples/get-started/csi_send.

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_mac.h"
#include "nvs_flash.h"

static const char *TAG = "csi_tx";

static const uint8_t BROADCAST_MAC[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

static void wifi_init(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    // Pin to a fixed channel so receivers can stay tuned.
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_CSI_TX_CHANNEL, WIFI_SECOND_CHAN_NONE));
}

static void espnow_init(void) {
    ESP_ERROR_CHECK(esp_now_init());
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
    uint8_t payload[16];
    TickType_t next = xTaskGetTickCount();
    while (1) {
        memcpy(payload, &seq, sizeof(seq));
        memset(payload + sizeof(seq), 0xA5, sizeof(payload) - sizeof(seq));
        esp_err_t err = esp_now_send(BROADCAST_MAC, payload, sizeof(payload));
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "esp_now_send: %s", esp_err_to_name(err));
        }
        seq++;
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
