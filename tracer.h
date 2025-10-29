#ifndef TRACER_H
#define TRACER_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

typedef struct {
    Vec3 albedo;
    float roughness;
    float metalness;
} Material;

typedef struct {
    Vec3 c0;
    Vec3 c1;
    Vec3 c2;
} Mat3;

typedef enum {
    OBJ_PLANE,
    OBJ_CUBE,
    OBJ_SPHERE,
    OBJ_CYLINDER,
    OBJ_TORUS
} PrimitiveType;

typedef struct {
    Vec3 normal;
    Vec3 point;
} PlaneData;

typedef struct {
    float half_extent;
} CubeData;

typedef struct {
    float radius;
} SphereData;

typedef struct {
    float radius;
    float height;
} CylinderData;

typedef struct {
    float major_radius;
    float minor_radius;
} TorusData;

typedef struct {
    PrimitiveType type;
    Vec3 position;
    Mat3 basis;
    Material material;
    union {
        PlaneData plane;
        CubeData cube;
        SphereData sphere;
        CylinderData cylinder;
        TorusData torus;
    } shape;
} Primitive;

typedef struct {
    Vec3 position;
    float intensity;
} Light;

#define MAX_PRIMITIVES 32

typedef struct {
    Primitive objects[MAX_PRIMITIVES];
    size_t object_count;
    Light light;
    Vec3 background;
} Scene;

typedef struct {
    float t;
    Vec3 position;
    Vec3 normal;
    Material material;
    const Primitive *primitive;
} HitRecord;

/* Vector helpers */
Vec3 vec3(float x, float y, float z);
Vec3 vec3_add(Vec3 a, Vec3 b);
Vec3 vec3_sub(Vec3 a, Vec3 b);
Vec3 vec3_mul(Vec3 a, Vec3 b);
Vec3 vec3_scale(Vec3 v, float s);
Vec3 vec3_clamp(Vec3 v, float min_val, float max_val);
Vec3 vec3_normalize(Vec3 v);
Vec3 vec3_cross(Vec3 a, Vec3 b);
float vec3_dot(Vec3 a, Vec3 b);
float vec3_length(Vec3 v);
float vec3_length_sq(Vec3 v);

/* Matrix helpers */
Mat3 mat3_identity(void);
Mat3 mat3_from_euler(float pitch, float yaw, float roll);
Mat3 mat3_transpose(Mat3 m);
Vec3 mat3_mul_vec3(Mat3 m, Vec3 v);

/* Scene helpers */
Primitive make_plane(Vec3 point, Vec3 normal, Material material);
Primitive make_cube(Vec3 center, Mat3 basis, float size, Material material);
Primitive make_sphere(Vec3 center, float radius, Material material);
Primitive make_cylinder(Vec3 center, Mat3 basis, float radius, float height, Material material);
Primitive make_torus(Vec3 center, Mat3 basis, float major_radius, float minor_radius, Material material);
Light make_light(Vec3 position, float intensity);

/* Material helpers */
Material material_diffuse(Vec3 color);
Material material_metal(Vec3 color, float roughness);

/* Ray tracing */
bool trace_scene(const Scene *scene, Ray ray, HitRecord *out_hit);
Vec3 shade_hit(const Scene *scene, Ray ray, const HitRecord *hit);
Vec3 trace_ray(const Scene *scene, Ray ray);

#endif /* TRACER_H */
