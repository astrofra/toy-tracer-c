#include "tracer.h"

#include <float.h>
#include <math.h>
#include <string.h>

#define PI 3.14159265358979323846f
#define EPSILON 1e-4f

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float radians(float degrees) {
    return degrees * (PI / 180.0f);
}

Vec3 vec3(float x, float y, float z) {
    Vec3 v = {x, y, z};
    return v;
}

Vec3 vec3_add(Vec3 a, Vec3 b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec3 vec3_mul(Vec3 a, Vec3 b) {
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

Vec3 vec3_scale(Vec3 v, float s) {
    return vec3(v.x * s, v.y * s, v.z * s);
}

Vec3 vec3_clamp(Vec3 v, float min_val, float max_val) {
    return vec3(clampf(v.x, min_val, max_val),
                clampf(v.y, min_val, max_val),
                clampf(v.z, min_val, max_val));
}

float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

float vec3_length_sq(Vec3 v) {
    return vec3_dot(v, v);
}

float vec3_length(Vec3 v) {
    return sqrtf(vec3_length_sq(v));
}

Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    if (len < EPSILON) {
        return vec3(0.0f, 0.0f, 0.0f);
    }
    return vec3_scale(v, 1.0f / len);
}

static Vec3 vec3_mix(Vec3 a, Vec3 b, float t) {
    return vec3_add(vec3_scale(a, 1.0f - t), vec3_scale(b, t));
}

Mat3 mat3_identity(void) {
    Mat3 m = {
        vec3(1.0f, 0.0f, 0.0f),
        vec3(0.0f, 1.0f, 0.0f),
        vec3(0.0f, 0.0f, 1.0f)
    };
    return m;
}

Mat3 mat3_from_euler(float pitch_deg, float yaw_deg, float roll_deg) {
    float pitch = radians(pitch_deg);
    float yaw = radians(yaw_deg);
    float roll = radians(roll_deg);

    float cp = cosf(pitch);
    float sp = sinf(pitch);
    float cy = cosf(yaw);
    float sy = sinf(yaw);
    float cr = cosf(roll);
    float sr = sinf(roll);

    Mat3 m;
    /* Compose Z (roll) -> X (pitch) -> Y (yaw) rotations for an intuitive scene setup */
    m.c0 = vec3(
        cy * cr + sy * sp * sr,
        cp * sr,
        -sy * cr + cy * sp * sr
    );
    m.c1 = vec3(
        -cy * sr + sy * sp * cr,
        cp * cr,
        sy * sr + cy * sp * cr
    );
    m.c2 = vec3(
        sy * cp,
        -sp,
        cy * cp
    );
    return m;
}

Mat3 mat3_transpose(Mat3 m) {
    Mat3 t;
    t.c0 = vec3(m.c0.x, m.c1.x, m.c2.x);
    t.c1 = vec3(m.c0.y, m.c1.y, m.c2.y);
    t.c2 = vec3(m.c0.z, m.c1.z, m.c2.z);
    return t;
}

Vec3 mat3_mul_vec3(Mat3 m, Vec3 v) {
    return vec3(
        m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z,
        m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z,
        m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z
    );
}

Primitive make_plane(Vec3 point, Vec3 normal, Material material) {
    Primitive p = {0};
    p.type = OBJ_PLANE;
    p.position = point;
    p.basis = mat3_identity();
    p.material = material;
    p.shape.plane.point = point;
    p.shape.plane.normal = vec3_normalize(normal);
    return p;
}

Primitive make_cube(Vec3 center, Mat3 basis, float size, Material material) {
    Primitive p = {0};
    p.type = OBJ_CUBE;
    p.position = center;
    p.basis = basis;
    p.material = material;
    p.shape.cube.half_extent = size * 0.5f;
    return p;
}

Primitive make_sphere(Vec3 center, float radius, Material material) {
    Primitive p = {0};
    p.type = OBJ_SPHERE;
    p.position = center;
    p.basis = mat3_identity();
    p.material = material;
    p.shape.sphere.radius = radius;
    return p;
}

Primitive make_cylinder(Vec3 center, Mat3 basis, float radius, float height, Material material) {
    Primitive p = {0};
    p.type = OBJ_CYLINDER;
    p.position = center;
    p.basis = basis;
    p.material = material;
    p.shape.cylinder.radius = radius;
    p.shape.cylinder.height = height;
    return p;
}

Primitive make_torus(Vec3 center, Mat3 basis, float major_radius, float minor_radius, Material material) {
    Primitive p = {0};
    p.type = OBJ_TORUS;
    p.position = center;
    p.basis = basis;
    p.material = material;
    p.shape.torus.major_radius = major_radius;
    p.shape.torus.minor_radius = minor_radius;
    return p;
}

Light make_light(Vec3 position, float intensity) {
    Light l;
    l.position = position;
    l.intensity = intensity;
    return l;
}

Material material_diffuse(Vec3 color) {
    Material m;
    m.albedo = color;
    m.roughness = 0.8f;
    m.metalness = 0.0f;
    return m;
}

Material material_metal(Vec3 color, float roughness) {
    Material m;
    m.albedo = color;
    m.roughness = clampf(roughness, 0.05f, 1.0f);
    m.metalness = 1.0f;
    return m;
}

static void to_local_ray(const Primitive *primitive, Ray ray, Vec3 *out_origin, Vec3 *out_dir) {
    Mat3 inverse = mat3_transpose(primitive->basis);
    Vec3 relative = vec3_sub(ray.origin, primitive->position);
    *out_origin = mat3_mul_vec3(inverse, relative);
    *out_dir = mat3_mul_vec3(inverse, ray.direction);
}

static bool intersect_plane(const Primitive *primitive, Ray ray, float *t_out, Vec3 *normal_out) {
    Vec3 normal = primitive->shape.plane.normal;
    float denom = vec3_dot(normal, ray.direction);
    if (fabsf(denom) < EPSILON) {
        return false;
    }
    float t = vec3_dot(vec3_sub(primitive->shape.plane.point, ray.origin), normal) / denom;
    if (t <= EPSILON) {
        return false;
    }
    *t_out = t;
    *normal_out = normal;
    return true;
}

static bool intersect_sphere(const Primitive *primitive, Ray ray, float *t_out, Vec3 *normal_out) {
    Vec3 oc = vec3_sub(ray.origin, primitive->position);
    float radius = primitive->shape.sphere.radius;
    float a = vec3_dot(ray.direction, ray.direction);
    float b = 2.0f * vec3_dot(oc, ray.direction);
    float c = vec3_dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    float sqrt_disc = sqrtf(discriminant);
    float inv_denom = 0.5f / a;
    float t0 = (-b - sqrt_disc) * inv_denom;
    float t1 = (-b + sqrt_disc) * inv_denom;
    float t = t0 > EPSILON ? t0 : t1;
    if (t <= EPSILON) {
        return false;
    }
    Vec3 hit_point = vec3_add(ray.origin, vec3_scale(ray.direction, t));
    Vec3 normal = vec3_normalize(vec3_sub(hit_point, primitive->position));
    *t_out = t;
    *normal_out = normal;
    return true;
}

static bool intersect_cube(const Primitive *primitive, Ray ray, float *t_out, Vec3 *normal_out) {
    Vec3 local_origin, local_dir;
    to_local_ray(primitive, ray, &local_origin, &local_dir);
    float half = primitive->shape.cube.half_extent;

    float t_min = -FLT_MAX;
    float t_max = FLT_MAX;

    for (int axis = 0; axis < 3; ++axis) {
        float origin_component = axis == 0 ? local_origin.x : (axis == 1 ? local_origin.y : local_origin.z);
        float dir_component = axis == 0 ? local_dir.x : (axis == 1 ? local_dir.y : local_dir.z);

        if (fabsf(dir_component) < EPSILON) {
            if (origin_component < -half || origin_component > half) {
                return false;
            }
            continue;
        }

        float inv = 1.0f / dir_component;
        float t0 = (-half - origin_component) * inv;
        float t1 = (half - origin_component) * inv;
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        if (t0 > t_min) {
            t_min = t0;
        }
        if (t1 < t_max) {
            t_max = t1;
        }
        if (t_min > t_max) {
            return false;
        }
    }

    float t_hit = t_min > EPSILON ? t_min : t_max;
    if (t_hit <= EPSILON) {
        return false;
    }

    Vec3 local_hit = vec3_add(local_origin, vec3_scale(local_dir, t_hit));
    Vec3 local_normal = vec3(0.0f, 0.0f, 0.0f);
    float abs_x = fabsf(local_hit.x);
    float abs_y = fabsf(local_hit.y);
    float abs_z = fabsf(local_hit.z);
    float max_abs = fmaxf(abs_x, fmaxf(abs_y, abs_z));
    if (fabsf(max_abs - abs_x) <= 1e-3f) {
        local_normal = vec3(local_hit.x > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
    } else if (fabsf(max_abs - abs_y) <= 1e-3f) {
        local_normal = vec3(0.0f, local_hit.y > 0.0f ? 1.0f : -1.0f, 0.0f);
    } else {
        local_normal = vec3(0.0f, 0.0f, local_hit.z > 0.0f ? 1.0f : -1.0f);
    }

    Vec3 world_normal = vec3_normalize(mat3_mul_vec3(primitive->basis, local_normal));
    *t_out = t_hit;
    *normal_out = world_normal;
    return true;
}

static bool solve_quadratic(float a, float b, float c, float *t0, float *t1) {
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    float sqrt_disc = sqrtf(discriminant);
    float inv_denom = 0.5f / a;
    *t0 = (-b - sqrt_disc) * inv_denom;
    *t1 = (-b + sqrt_disc) * inv_denom;
    if (*t0 > *t1) {
        float tmp = *t0;
        *t0 = *t1;
        *t1 = tmp;
    }
    return true;
}

static bool intersect_cylinder(const Primitive *primitive, Ray ray, float *t_out, Vec3 *normal_out) {
    Vec3 local_origin, local_dir;
    to_local_ray(primitive, ray, &local_origin, &local_dir);
    float radius = primitive->shape.cylinder.radius;
    float half_height = primitive->shape.cylinder.height * 0.5f;

    float a = local_dir.x * local_dir.x + local_dir.z * local_dir.z;
    float b = 2.0f * (local_origin.x * local_dir.x + local_origin.z * local_dir.z);
    float c = local_origin.x * local_origin.x + local_origin.z * local_origin.z - radius * radius;

    float best_t = FLT_MAX;
    Vec3 best_normal = vec3(0.0f, 0.0f, 0.0f);

    if (fabsf(a) > EPSILON) {
        float t0, t1;
        if (solve_quadratic(a, b, c, &t0, &t1)) {
            float t_candidates[2] = {t0, t1};
            for (int i = 0; i < 2; ++i) {
                float t = t_candidates[i];
                if (t <= EPSILON || t >= best_t) {
                    continue;
                }
                float y = local_origin.y + t * local_dir.y;
                if (y < -half_height - EPSILON || y > half_height + EPSILON) {
                    continue;
                }
                Vec3 local_point = vec3_add(local_origin, vec3_scale(local_dir, t));
                Vec3 side_normal = vec3_normalize(vec3(local_point.x, 0.0f, local_point.z));
                best_t = t;
                best_normal = mat3_mul_vec3(primitive->basis, side_normal);
            }
        }
    }

    for (int cap = -1; cap <= 1; cap += 2) {
        float plane_y = cap * half_height;
        float denom = local_dir.y;
        if (fabsf(denom) < EPSILON) {
            continue;
        }
        float t = (plane_y - local_origin.y) / denom;
        if (t <= EPSILON || t >= best_t) {
            continue;
        }
        Vec3 local_point = vec3_add(local_origin, vec3_scale(local_dir, t));
        if (local_point.x * local_point.x + local_point.z * local_point.z > radius * radius + EPSILON) {
            continue;
        }
        Vec3 cap_normal = mat3_mul_vec3(primitive->basis, vec3(0.0f, (float)cap, 0.0f));
        best_t = t;
        best_normal = cap_normal;
    }

    if (best_t >= FLT_MAX) {
        return false;
    }

    *t_out = best_t;
    *normal_out = vec3_normalize(best_normal);
    return true;
}

static double evaluate_quartic(const double coeffs[5], double t) {
    return (((coeffs[4] * t + coeffs[3]) * t + coeffs[2]) * t + coeffs[1]) * t + coeffs[0];
}

static bool find_bracket(const double coeffs[5], double *t_low, double *t_high) {
    double f_prev = evaluate_quartic(coeffs, *t_low);
    double step = 0.25;
    for (double t = *t_low + step; t <= *t_high; t += step) {
        double f_curr = evaluate_quartic(coeffs, t);
        if (fabs(f_curr) < 1e-9) {
            *t_low = t;
            *t_high = t;
            return true;
        }
        if (f_prev * f_curr < 0.0) {
            *t_low = t - step;
            *t_high = t;
            return true;
        }
        f_prev = f_curr;
    }
    return false;
}

/* Torus intersection uses a numeric bisection to keep the algebra approachable for students. */
static bool solve_quartic_numeric(const double coeffs[5], double max_t, double *root_out) {
    double t_low = EPSILON;
    double t_high = max_t;
    if (!find_bracket(coeffs, &t_low, &t_high)) {
        return false;
    }
    if (fabs(t_low - t_high) < 1e-6) {
        *root_out = t_low;
        return true;
    }
    double f_low = evaluate_quartic(coeffs, t_low);
    double f_high = evaluate_quartic(coeffs, t_high);
    if (fabs(f_low) < 1e-9) {
        *root_out = t_low;
        return true;
    }
    if (fabs(f_high) < 1e-9) {
        *root_out = t_high;
        return true;
    }
    for (int i = 0; i < 64; ++i) {
        double mid = 0.5 * (t_low + t_high);
        double f_mid = evaluate_quartic(coeffs, mid);
        if (fabs(f_mid) < 1e-9 || fabs(t_high - t_low) < 1e-5) {
            *root_out = mid;
            return true;
        }
        if (f_low * f_mid < 0.0) {
            t_high = mid;
            f_high = f_mid;
        } else {
            t_low = mid;
            f_low = f_mid;
        }
    }
    *root_out = 0.5 * (t_low + t_high);
    return true;
}

static bool intersect_torus(const Primitive *primitive, Ray ray, float *t_out, Vec3 *normal_out) {
    Vec3 local_origin, local_dir;
    to_local_ray(primitive, ray, &local_origin, &local_dir);

    double dx = local_dir.x;
    double dy = local_dir.y;
    double dz = local_dir.z;
    double ox = local_origin.x;
    double oy = local_origin.y;
    double oz = local_origin.z;

    double R = primitive->shape.torus.major_radius;
    double r = primitive->shape.torus.minor_radius;

    double sum_dir_sq = dx * dx + dy * dy + dz * dz;
    double e = ox * ox + oy * oy + oz * oz - (R * R + r * r);
    double f = ox * dx + oy * dy + oz * dz;
    double four_R_sq = 4.0 * R * R;

    double coeffs[5];
    coeffs[4] = sum_dir_sq * sum_dir_sq;
    coeffs[3] = 4.0 * sum_dir_sq * f;
    coeffs[2] = 2.0 * sum_dir_sq * e + 4.0 * f * f + four_R_sq * dy * dy;
    coeffs[1] = 4.0 * f * e + 2.0 * four_R_sq * dy * oy;
    coeffs[0] = e * e + four_R_sq * (oy * oy - r * r);

    double root;
    if (!solve_quartic_numeric(coeffs, 200.0, &root)) {
        return false;
    }

    if (root <= EPSILON) {
        return false;
    }

    Vec3 local_hit = vec3_add(local_origin, vec3_scale(local_dir, (float)root));
    double sum = local_hit.x * local_hit.x + local_hit.y * local_hit.y + local_hit.z * local_hit.z;
    double common = sum + R * R - r * r;

    Vec3 gradient = vec3(
        (float)(4.0 * local_hit.x * common - 8.0 * R * R * local_hit.x),
        (float)(4.0 * local_hit.y * common),
        (float)(4.0 * local_hit.z * common - 8.0 * R * R * local_hit.z)
    );
    Vec3 world_normal = vec3_normalize(mat3_mul_vec3(primitive->basis, gradient));
    *t_out = (float)root;
    *normal_out = world_normal;
    return true;
}

static bool intersect_primitive(const Primitive *primitive, Ray ray, float *t_out, Vec3 *normal_out) {
    switch (primitive->type) {
    case OBJ_PLANE:
        return intersect_plane(primitive, ray, t_out, normal_out);
    case OBJ_SPHERE:
        return intersect_sphere(primitive, ray, t_out, normal_out);
    case OBJ_CUBE:
        return intersect_cube(primitive, ray, t_out, normal_out);
    case OBJ_CYLINDER:
        return intersect_cylinder(primitive, ray, t_out, normal_out);
    case OBJ_TORUS:
        return intersect_torus(primitive, ray, t_out, normal_out);
    default:
        return false;
    }
}

bool trace_scene(const Scene *scene, Ray ray, HitRecord *out_hit) {
    float closest = FLT_MAX;
    bool hit = false;
    HitRecord record = {0};
    for (size_t i = 0; i < scene->object_count; ++i) {
        const Primitive *primitive = &scene->objects[i];
        float t;
        Vec3 normal;
        if (!intersect_primitive(primitive, ray, &t, &normal)) {
            continue;
        }
        if (t < EPSILON || t >= closest) {
            continue;
        }
        closest = t;
        record.t = t;
        record.position = vec3_add(ray.origin, vec3_scale(ray.direction, t));
        record.normal = vec3_normalize(normal);
        record.material = primitive->material;
        record.primitive = primitive;
        hit = true;
    }
    if (hit && out_hit) {
        *out_hit = record;
    }
    return hit;
}

static bool is_shadowed(const Scene *scene, Vec3 point, Vec3 to_light, float max_distance, const Primitive *self) {
    Ray shadow_ray;
    shadow_ray.origin = vec3_add(point, vec3_scale(to_light, EPSILON));
    shadow_ray.direction = to_light;
    float closest = max_distance;
    for (size_t i = 0; i < scene->object_count; ++i) {
        const Primitive *primitive = &scene->objects[i];
        if (primitive == self) {
            continue;
        }
        float t;
        Vec3 normal;
        if (!intersect_primitive(primitive, shadow_ray, &t, &normal)) {
            continue;
        }
        if (t > EPSILON && t < closest) {
            return true;
        }
    }
    return false;
}

static float fresnel_schlick(float cos_theta, float base) {
    return base + (1.0f - base) * powf(1.0f - cos_theta, 5.0f);
}

static float distribution_ggx(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom + EPSILON);
}

static float geometry_schlick_ggx(float NdotV, float k) {
    return NdotV / (NdotV * (1.0f - k) + k + EPSILON);
}

static float geometry_smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    float g1 = geometry_schlick_ggx(NdotV, k);
    float g2 = geometry_schlick_ggx(NdotL, k);
    return g1 * g2;
}

Vec3 shade_hit(const Scene *scene, Ray ray, const HitRecord *hit) {
    Vec3 N = hit->normal;
    Vec3 L = vec3_sub(scene->light.position, hit->position);
    float distance = vec3_length(L);
    if (distance < EPSILON) {
        return hit->material.albedo;
    }
    Vec3 light_dir = vec3_scale(L, 1.0f / distance);
    Vec3 V = vec3_normalize(vec3_scale(ray.direction, -1.0f));
    Vec3 H = vec3_normalize(vec3_add(light_dir, V));

    float NdotL = clampf(vec3_dot(N, light_dir), 0.0f, 1.0f);
    float NdotV = clampf(vec3_dot(N, V), 0.0f, 1.0f);
    float NdotH = clampf(vec3_dot(N, H), 0.0f, 1.0f);
    float VdotH = clampf(vec3_dot(V, H), 0.0f, 1.0f);

    float attenuation = scene->light.intensity / (distance * distance + 1.0f);

    if (is_shadowed(scene, hit->position, light_dir, distance - EPSILON, hit->primitive)) {
        attenuation *= 0.1f;
    }

    Vec3 F0_dielectric = vec3(0.04f, 0.04f, 0.04f);
    Vec3 F0 = vec3_mix(F0_dielectric, hit->material.albedo, hit->material.metalness);

    Vec3 F = vec3(
        fresnel_schlick(VdotH, F0.x),
        fresnel_schlick(VdotH, F0.y),
        fresnel_schlick(VdotH, F0.z)
    );

    float alpha = hit->material.roughness * hit->material.roughness;
    float D = distribution_ggx(NdotH, alpha);
    float G = geometry_smith(NdotV, NdotL, hit->material.roughness);

    float denom = 4.0f * NdotV * NdotL + EPSILON;
    Vec3 specular = vec3_scale(F, (D * G) / denom);

    Vec3 kd = vec3_scale(vec3_sub(vec3(1.0f, 1.0f, 1.0f), F), 1.0f - hit->material.metalness);
    Vec3 diffuse = vec3_scale(vec3_mul(hit->material.albedo, kd), 1.0f / PI);

    Vec3 color = vec3_add(diffuse, specular);
    color = vec3_scale(color, NdotL * attenuation);

    Vec3 ambient = vec3_scale(hit->material.albedo, 0.05f);
    color = vec3_add(color, ambient);
    return vec3_clamp(color, 0.0f, 1.0f);
}

Vec3 trace_ray(const Scene *scene, Ray ray) {
    HitRecord hit;
    if (trace_scene(scene, ray, &hit)) {
        return shade_hit(scene, ray, &hit);
    }
    return scene->background;
}
