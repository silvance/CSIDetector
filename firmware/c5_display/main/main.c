// CSI Detector remote display.
//
// Joins the host's hotspot and listens on a UDP port for state packets
// emitted by `python run.py publish`. Each packet is 24 bytes:
//
//     uint8   version          (= 1)
//     uint8   state             (0=INIT, 1=EMPTY, 2=MOTION)
//     uint8   reserved[2]
//     float32 median_ratio
//     float32 max_ratio
//     uint32  state_changes
//     uint64  ts_ms
//
// The display rendering is a stub — call display_show_state() with the
// new state and let the LCD code (TBD; depends on the NM-CYD-C5's
// internal LCD pinout, which we don't have yet) paint a screen.

#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"

static const char *TAG = "c5_disp";

typedef enum {
    PRESENCE_INIT = 0,
    PRESENCE_EMPTY = 1,
    PRESENCE_MOTION = 2,
} presence_state_t;

typedef struct __attribute__((packed)) {
    uint8_t  version;
    uint8_t  state;
    uint8_t  reserved[2];
    float    median_ratio;
    float    max_ratio;
    uint32_t state_changes;
    uint64_t ts_ms;
} state_pkt_t;
_Static_assert(sizeof(state_pkt_t) == 24, "state_pkt_t must be 24 bytes");


// ---- display stub --------------------------------------------------
// Replace these with calls to the actual LCD driver once we know the
// NM-CYD-C5's internal LCD pinout. For now, log the state to UART so
// the rest of the pipeline can be exercised before the screen works.

static void display_init(void) {
    ESP_LOGI(TAG, "display_init: stub — replace with LCD driver init "
                  "(SPI bus, panel handle, backlight on, clear)");
}

static void display_show_state(presence_state_t state, float median, float mx) {
    static const char *names[] = {"INIT", "EMPTY", "MOTION"};
    static const char *colors[] = {"GRAY", "GREEN", "RED"};
    if (state > PRESENCE_MOTION) state = PRESENCE_INIT;
    ESP_LOGI(TAG, "show_state: %s (%s)  median=%.2fx  max=%.2fx",
             names[state], colors[state], median, mx);
    // TODO: paint background `colors[state]`, render text `names[state]`
    // and the median/max numbers below it.
}


// ---- wifi ----------------------------------------------------------

static void wifi_event(void *arg, esp_event_base_t base,
                       int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *e = data;
        ESP_LOGW(TAG, "STA disconnected (reason=%d), reconnecting",
                 e ? e->reason : 0);
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
        ESP_LOGI(TAG, "STA associated to AP");
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = data;
        ESP_LOGI(TAG, "got IP: " IPSTR, IP2STR(&e->ip_info.ip));
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

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event, NULL, NULL));

    wifi_config_t sta = {0};
    strlcpy((char *)sta.sta.ssid, CONFIG_C5_WIFI_SSID, sizeof(sta.sta.ssid));
    strlcpy((char *)sta.sta.password, CONFIG_C5_WIFI_PASS, sizeof(sta.sta.password));
    sta.sta.channel = CONFIG_C5_WIFI_CHANNEL;
    sta.sta.scan_method = WIFI_FAST_SCAN;
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_connect());
}


// ---- udp listener --------------------------------------------------

static void udp_task(void *arg) {
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket(): errno %d", errno);
        vTaskDelete(NULL);
        return;
    }
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &yes, sizeof(yes));

    struct sockaddr_in bind_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(CONFIG_C5_LISTEN_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    if (bind(sock, (struct sockaddr *)&bind_addr, sizeof(bind_addr)) < 0) {
        ESP_LOGE(TAG, "bind(): errno %d", errno);
        close(sock);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "listening on UDP :%d", CONFIG_C5_LISTEN_PORT);

    presence_state_t last_state = PRESENCE_INIT - 1;  // force first render
    state_pkt_t pkt;
    while (1) {
        ssize_t n = recv(sock, &pkt, sizeof(pkt), 0);
        if (n != (ssize_t)sizeof(pkt)) {
            if (n < 0) {
                ESP_LOGW(TAG, "recv(): errno %d", errno);
                vTaskDelay(pdMS_TO_TICKS(200));
            }
            continue;
        }
        if (pkt.version != 1) continue;
        presence_state_t s = (presence_state_t)pkt.state;
        // Always update the on-screen ratios; only repaint backgrounds
        // on actual state transition (saves the LCD some flicker).
        if (s != last_state) {
            ESP_LOGI(TAG, "state %d -> %d  (median %.2fx)",
                     last_state, s, pkt.median_ratio);
            last_state = s;
        }
        display_show_state(s, pkt.median_ratio, pkt.max_ratio);
    }
}


void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    display_init();
    wifi_init();
    xTaskCreate(udp_task, "udp", 4096, NULL, 5, NULL);
}
