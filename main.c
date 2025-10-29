#include "scene.h"
#include "tracer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static float to_radians(float degrees) {
    const float pi = 3.14159265358979323846f;
    return degrees * (pi / 180.0f);
}

static uint32_t pack_color(Vec3 color) {
    Vec3 clamped = vec3_clamp(color, 0.0f, 1.0f);
    /* Simple gamma compensation keeps the image from looking too dark on default viewers. */
    float gamma = 1.0f / 2.2f;
    uint8_t r = (uint8_t)(powf(clamped.x, gamma) * 255.0f);
    uint8_t g = (uint8_t)(powf(clamped.y, gamma) * 255.0f);
    uint8_t b = (uint8_t)(powf(clamped.z, gamma) * 255.0f);
    return (uint32_t)b | ((uint32_t)g << 8u) | ((uint32_t)r << 16u) | (0xFFu << 24u);
}

static int write_tga(const char *filename, const uint32_t *pixels, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return 0;
    }

    unsigned char header[18] = {0};
    header[2] = 2; /* Uncompressed true-color image */
    header[12] = (unsigned char)(width & 0xFF);
    header[13] = (unsigned char)((width >> 8) & 0xFF);
    header[14] = (unsigned char)(height & 0xFF);
    header[15] = (unsigned char)((height >> 8) & 0xFF);
    header[16] = 32; /* Bits per pixel */
    header[17] = 0x20; /* Bit 5 set => top-left origin aligns with framebuffer indexing */

    fwrite(header, sizeof(header), 1, fp);
    fwrite(pixels, sizeof(uint32_t), (size_t)(width * height), fp);
    fclose(fp);
    return 1;
}

int main(void) {
    const int width = 640;
    const int height = 480;
    const float vertical_fov = 60.0f;

    Scene scene;
    setup_scene(&scene);

    size_t pixel_count = (size_t)width * (size_t)height;
    uint32_t *framebuffer = (uint32_t *)malloc(pixel_count * sizeof(uint32_t));
    if (!framebuffer) {
        fprintf(stderr, "Failed to allocate framebuffer (%d x %d)\n", width, height);
        return EXIT_FAILURE;
    }

    Vec3 camera_origin = vec3(0.0f, 0.0f, 0.0f);
    float aspect = (float)width / (float)height;
    float scale = tanf(to_radians(vertical_fov * 0.5f));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float ndc_x = ((x + 0.5f) / (float)width) * 2.0f - 1.0f;
            float ndc_y = 1.0f - ((y + 0.5f) / (float)height) * 2.0f;
            Vec3 ray_dir = vec3(ndc_x * aspect * scale, ndc_y * scale, 1.0f);
            Ray ray = {camera_origin, vec3_normalize(ray_dir)};
            Vec3 color = trace_ray(&scene, ray);
            framebuffer[y * width + x] = pack_color(color);
        }
    }

    if (!write_tga("output.tga", framebuffer, width, height)) {
        fprintf(stderr, "Failed to write output.tga\n");
        free(framebuffer);
        return EXIT_FAILURE;
    }

    printf("Rendered image written to output.tga (%dx%d)\n", width, height);

    free(framebuffer);
    return EXIT_SUCCESS;
}
