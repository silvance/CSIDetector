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
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "esp_heap_caps.h"

static const char *TAG = "c5_disp";

// NM-CYD-C5 LCD pinout (from RockBase-iot/NM-CYD-C5
// Demos/Arduino/libraries/TFT_eSPI/User_Setup-NM-CYD-C5.h).
// ST7789 driver, 320×240, SPI shared with the SD slot (CS=10) and the
// touch chip (CS=1). Backlight is active-high on GPIO25.
#define LCD_HOST            SPI2_HOST
#define LCD_PIN_MISO        2
#define LCD_PIN_MOSI        7
#define LCD_PIN_SCLK        6
#define LCD_PIN_CS          23
#define LCD_PIN_DC          24
#define LCD_PIN_RST         (-1)
#define LCD_PIN_BL          25
#define LCD_H_RES           240
#define LCD_V_RES           320
#define LCD_PIXEL_CLOCK_HZ  (20 * 1000 * 1000)
#define LCD_CMD_BITS        8
#define LCD_PARAM_BITS      8
#define LCD_BIT_DEPTH       16

static esp_lcd_panel_handle_t s_panel = NULL;

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


// ---- display -------------------------------------------------------

// Solid-color background per state. RGB565 (big-endian on the wire for
// st7789's standard mode). Color alone reads from across the room;
// text can be added later if demo polish demands it.
//   INIT   = dark grey  (boot / no data yet)
//   EMPTY  = green       (room is unoccupied)
//   MOTION = red         (motion detected)
static const uint16_t STATE_COLOR_RGB565[] = {
    0x4208,  // (#404040) dark grey
    0x07E0,  // (#00FF00) green
    0xF800,  // (#FF0000) red
};

// Fill buffer reused across frames. 320×240×2 bytes = 150 KiB; the
// ST7789 happily accepts a single big draw_bitmap call as long as the
// buffer is in DMA-capable RAM. Allocated once.
static uint16_t *s_fb = NULL;

static void display_init(void) {
    // Backlight off-then-on so the screen doesn't flash garbage during
    // panel init.
    gpio_config_t bl_cfg = {
        .pin_bit_mask = 1ULL << LCD_PIN_BL,
        .mode = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&bl_cfg));
    gpio_set_level(LCD_PIN_BL, 0);

    spi_bus_config_t bus = {
        .miso_io_num = LCD_PIN_MISO,
        .mosi_io_num = LCD_PIN_MOSI,
        .sclk_io_num = LCD_PIN_SCLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = LCD_H_RES * LCD_V_RES * 2 + 8,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &bus, SPI_DMA_CH_AUTO));

    esp_lcd_panel_io_handle_t io = NULL;
    esp_lcd_panel_io_spi_config_t io_config = {
        .dc_gpio_num = LCD_PIN_DC,
        .cs_gpio_num = LCD_PIN_CS,
        .pclk_hz = LCD_PIXEL_CLOCK_HZ,
        .lcd_cmd_bits = LCD_CMD_BITS,
        .lcd_param_bits = LCD_PARAM_BITS,
        .spi_mode = 0,
        .trans_queue_depth = 10,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi(
        (esp_lcd_spi_bus_handle_t)LCD_HOST, &io_config, &io));

    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = LCD_PIN_RST,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB,
        .bits_per_pixel = LCD_BIT_DEPTH,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io, &panel_config, &s_panel));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(s_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(s_panel));
    // ST7789 ships with inverted colors enabled by hardware; flip back.
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(s_panel, true));
    // Landscape (320 wide × 240 tall) feels right for a wall display.
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(s_panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(s_panel, true, false));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(s_panel, true));

    s_fb = heap_caps_malloc(LCD_V_RES * LCD_H_RES * sizeof(uint16_t),
                             MALLOC_CAP_DMA);
    assert(s_fb != NULL);

    // Show INIT colour now so the user sees something during WiFi join.
    for (int i = 0; i < LCD_V_RES * LCD_H_RES; i++) {
        s_fb[i] = STATE_COLOR_RGB565[PRESENCE_INIT];
    }
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(s_panel, 0, 0,
                                              LCD_V_RES, LCD_H_RES, s_fb));
    gpio_set_level(LCD_PIN_BL, 1);   // backlight on now that buffer's clean
    ESP_LOGI(TAG, "display init: ST7789 %dx%d on SPI%d (MOSI=%d SCLK=%d CS=%d DC=%d BL=%d)",
             LCD_V_RES, LCD_H_RES, LCD_HOST,
             LCD_PIN_MOSI, LCD_PIN_SCLK, LCD_PIN_CS, LCD_PIN_DC, LCD_PIN_BL);
}

static void display_show_state(presence_state_t state, float median, float mx) {
    static const char *names[] = {"INIT", "EMPTY", "MOTION"};
    static presence_state_t last_painted = (presence_state_t)0xff;
    if (state > PRESENCE_MOTION) state = PRESENCE_INIT;

    if (state != last_painted) {
        ESP_LOGI(TAG, "paint: %s  median=%.2fx  max=%.2fx",
                 names[state], median, mx);
        const uint16_t color = STATE_COLOR_RGB565[state];
        // SPI sends 16-bit pixels MSB-first; ST7789 expects big-endian
        // RGB565, so swap on the way into the buffer.
        const uint16_t be = (color << 8) | (color >> 8);
        for (int i = 0; i < LCD_V_RES * LCD_H_RES; i++) s_fb[i] = be;
        if (s_panel) {
            esp_lcd_panel_draw_bitmap(s_panel, 0, 0,
                                      LCD_V_RES, LCD_H_RES, s_fb);
        }
        last_painted = state;
    }
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
