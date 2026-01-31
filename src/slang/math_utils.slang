// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// =============================================================================
// Math Utilities for Differentiable Volume Rendering
// =============================================================================
//
// Numerical stability patterns based on instant-ngp implementation:
// - Safe division with proper epsilon handling
// - Clamped exponentials to prevent overflow
// - Log-space operations for better precision
// - Gradient-safe implementations with custom backward derivatives
// =============================================================================

// =============================================================================
// Constants
// =============================================================================

// Threshold for "effectively zero" in divisions
// Following instant-ngp pattern: use small positive threshold
static const float TINY_VAL = 1e-6f;

// Clamp exp() input to avoid overflow
// instant-ngp uses -10 to 10 for RGB, -15 to 15 for density derivatives
// Using 15.0f as a balanced limit (exp(15) ≈ 3.3e6, exp(-15) ≈ 3e-7)
static const float EXP_LIMIT = 15.0f;

// Stricter limit for derivative computations to ensure gradient stability
static const float EXP_LIMIT_DERIVATIVE = 15.0f;

// Minimum transmittance threshold for early ray termination
// instant-ngp default: 0.01f (stop when 99% of light is blocked)
static const float MIN_TRANSMITTANCE = 0.01f;

// Epsilon for alpha comparisons (instant-ngp uses 1e-4f)
static const float ALPHA_EPSILON = 1e-4f;

// Minimum optical thickness for occupancy (instant-ngp: 0.01f)
static const float MIN_OPTICAL_THICKNESS = 0.01f;

// ln(2) constant for log-space exposure calculations
static const float LN2 = 0.6931471805599453f;

// log(2) threshold for log1mexp branch selection
static const float LOG2 = 0.6931471805599453f;

// =============================================================================
// Clipping & Clamping
// =============================================================================

[Differentiable]
float clip(float v, float minv, float maxv) {
    return max(min(v, maxv), minv);
}

[Differentiable]
float3 clip(float3 v, float minv, float maxv) {
    return float3(
        clip(v.x, minv, maxv),
        clip(v.y, minv, maxv),
        clip(v.z, minv, maxv)
    );
}

[Differentiable]
float4 clip(float4 v, float minv, float maxv) {
    return float4(
        clip(v.x, minv, maxv),
        clip(v.y, minv, maxv),
        clip(v.z, minv, maxv),
        clip(v.w, minv, maxv)
    );
}

// =============================================================================
// Safe Division
// =============================================================================
//
// Pattern from instant-ngp/src/testbed_nerf.cu:
//   float local_divisor = divisor[i];
//   inout[i] = local_divisor > 0.0f ? (inout[i] / local_divisor) : 0.0f;
//
// For general use with signed divisors, we check |b| < TINY_VAL
// =============================================================================

void bw_safe_div(inout DifferentialPair<float> a, inout DifferentialPair<float> b, float.Differential R) {
    // When denominator is too small, zero out gradients to prevent explosion
    // This is crucial for gradient stability in volume rendering
    if (abs(b.p) < TINY_VAL) {
        a = DifferentialPair<float>(a.p, 0.0f);
        b = DifferentialPair<float>(b.p, 0.0f);
    } else {
        // Standard quotient rule: d(a/b)/da = 1/b, d(a/b)/db = -a/b²
        float inv_b = 1.0f / b.p;
        a = DifferentialPair<float>(a.p, R * inv_b);
        b = DifferentialPair<float>(b.p, -a.p * inv_b * inv_b * R);
    }
}

[BackwardDerivative(bw_safe_div)]
float safe_div(in float a, in float b) {
    return abs(b) < TINY_VAL ? 0.0f : a / b;
}

// Variant following instant-ngp pattern (positive divisor only)
float safe_div_positive(in float a, in float b) {
    return b > 0.0f ? a / b : 0.0f;
}

[Differentiable]
float3 safe_div(in float3 a, in float b) {
    return float3(safe_div(a.x, b), safe_div(a.y, b), safe_div(a.z, b));
}

[Differentiable]
float4 safe_div(in float4 a, in float b) {
    return float4(safe_div(a.x, b), safe_div(a.y, b), safe_div(a.z, b), safe_div(a.w, b));
}

[Differentiable]
float3 safe_div(in float3 a, in float3 b) {
    return float3(safe_div(a.x, b.x), safe_div(a.y, b.y), safe_div(a.z, b.z));
}

// Safe division with custom fallback value (useful for color/density ratios)
// Pattern: fmaxf(1e-6f, denominator) from instant-ngp
[Differentiable]
float safe_div_with_min(in float a, in float b, in float min_denom = 1e-6f) {
    return a / max(abs(b), min_denom);
}

[Differentiable]
float3 safe_div_with_min(in float3 a, in float b, in float min_denom = 1e-6f) {
    return a / max(abs(b), min_denom);
}

// =============================================================================
// Safe Square Root & Normalization
// =============================================================================

void bw_safe_sqrt(inout DifferentialPair<float> a, float.Differential R) {
    if (a.p < TINY_VAL) {
        // Gradient is undefined at 0; return 0 to prevent explosion
        a = DifferentialPair<float>(a.p, 0.0f);
    } else {
        // d(sqrt(a))/da = 0.5 / sqrt(a)
        a = DifferentialPair<float>(a.p, R * 0.5f / sqrt(a.p));
    }
}

[BackwardDerivative(bw_safe_sqrt)]
float safe_sqrt(in float a) {
    return a < TINY_VAL ? 0.0f : sqrt(a);
}

[Differentiable]
float3 l2_normalize(float3 x) {
    float len_sq = dot(x, x);
    // Use max to ensure non-zero denominator
    return x * rsqrt(max(len_sq, TINY_VAL));
}

[Differentiable]
float4 l2_normalize(float4 x) {
    float len_sq = dot(x, x);
    return x * rsqrt(max(len_sq, TINY_VAL));
}

// =============================================================================
// Safe Exponential Functions
// =============================================================================
//
// Pattern from instant-ngp/nerf_device.cuh:
//   return expf(clamp(val, -10.0f, 10.0f));  // for RGB
//   return expf(clamp(val, -15.0f, 15.0f));  // for density derivatives
// =============================================================================

[Differentiable]
float safe_exp(float v) {
    return exp(clamp(v, -EXP_LIMIT, EXP_LIMIT));
}

// Stricter clamping for derivative computations
[Differentiable]
float safe_exp_derivative(float v) {
    return exp(clamp(v, -EXP_LIMIT_DERIVATIVE, EXP_LIMIT_DERIVATIVE));
}

// =============================================================================
// Safe Logarithm Functions
// =============================================================================

void bw_log(inout DifferentialPair<float> v, float.Differential R) {
    if (abs(v.p) < TINY_VAL) {
        // Gradient would explode; return 0
        v = DifferentialPair<float>(v.p, 0.0f);
    } else {
        // d(log(v))/dv = 1/v
        v = DifferentialPair<float>(v.p, R / v.p);
    }
}

[BackwardDerivative(bw_log)]
float safe_log(float v) {
    return log(max(v, TINY_VAL));
}

// =============================================================================
// log1p: log(1 + x) - numerically stable for small x
// =============================================================================

void bw_log1p(inout DifferentialPair<float> v, float.Differential R) {
    // d(log(1+v))/dv = 1/(1+v)
    float denom = 1.0f + v.p;
    if (abs(denom) < TINY_VAL) {
        v = DifferentialPair<float>(v.p, 0.0f);
    } else {
        v = DifferentialPair<float>(v.p, R / denom);
    }
}

[BackwardDerivative(bw_log1p)]
float _log1p(float v) {
    __intrinsic_asm "log1p(($0))";
}

// =============================================================================
// expm1: exp(x) - 1 - numerically stable for small x
// =============================================================================
//
// Critical for alpha calculation: alpha = 1 - exp(-tau) = -expm1(-tau)
// This avoids catastrophic cancellation when tau is small
// =============================================================================

void bw_expm1(inout DifferentialPair<float> v, float.Differential R) {
    // d(exp(v)-1)/dv = exp(v)
    // Use clamped exp for gradient stability
    v = DifferentialPair<float>(v.p, safe_exp(v.p) * R);
}

[BackwardDerivative(bw_expm1)]
float safe_expm1(float v) {
    // Clamp input to prevent overflow
    float clamped = clamp(v, -EXP_LIMIT, EXP_LIMIT);
    __intrinsic_asm "expm1($0)" : clamped;
}

// =============================================================================
// log1mexp: log(1 - exp(-x)) for x > 0
// =============================================================================
//
// Used for accumulating log-transmittance: logT_new = logT_old + log1mexp(tau)
// Different formulas used for numerical stability:
//   - x > log(2): use log1p(-exp(-x))  [avoids exp overflow]
//   - x <= log(2): use log(-expm1(-x)) [avoids catastrophic cancellation]
// =============================================================================

void bw_log1mexp(inout DifferentialPair<float> v, float.Differential R) {
    // d(log(1 - exp(-x)))/dx = exp(-x) / (1 - exp(-x))
    //                        = 1 / (exp(x) - 1)
    //                        = 1 / expm1(x)
    float expm1_v = safe_expm1(v.p);
    if (abs(expm1_v) < TINY_VAL) {
        v = DifferentialPair<float>(v.p, 0.0f);
    } else {
        v = DifferentialPair<float>(v.p, R / expm1_v);
    }
}

[BackwardDerivative(bw_log1mexp)]
float log1mexp(float x) {
    // Threshold from instant-ngp volume rendering patterns
    if (x > LOG2) {
        // For large x, exp(-x) is small, so use log1p for precision
        return _log1p(-safe_exp(-x));
    } else {
        // For small x, use expm1 to avoid cancellation
        float arg = -safe_expm1(-x);
        return safe_log(arg);
    }
}

// =============================================================================
// Alpha Calculation (Beer-Lambert Law)
// =============================================================================
//
// Pattern from instant-ngp:
//   float alpha = 1.f - __expf(-network_to_density(nerf_out.w, density_activation) * dt);
//   float weight = alpha * (1.0f - color.a);
//
// For differentiable rendering with better numerical properties:
//   alpha = -expm1(-tau) where tau = sigma * dt
// =============================================================================

// Calculate alpha from optical depth tau = sigma * dt
// Returns value in [0, 1], gradient-stable for small tau
[Differentiable]
float tau_to_alpha(float tau) {
    // Clamp tau to non-negative (negative density is unphysical)
    tau = max(tau, 0.0f);
    // alpha = 1 - exp(-tau) = -expm1(-tau)
    return -safe_expm1(-tau);
}

// Calculate weight for compositing: weight = alpha * T
// where T = 1 - accumulated_alpha (transmittance)
[Differentiable]
float composite_weight(float alpha, float accumulated_alpha) {
    float T = 1.0f - accumulated_alpha;
    return alpha * T;
}

// Check if ray should terminate (transmittance below threshold)
// Pattern from instant-ngp: color.a > (1.0f - min_transmittance)
bool should_terminate(float accumulated_alpha) {
    return accumulated_alpha > (1.0f - MIN_TRANSMITTANCE);
}

bool should_terminate(float accumulated_alpha, float threshold) {
    return accumulated_alpha > (1.0f - threshold);
}

// =============================================================================
// Quaternion Operations
// =============================================================================

[Differentiable]
float4 quat_conjugate(float4 q) {
    return float4(q.x, -q.yzw);
}

/// Rotate vector by quaternion q (q = [w, x, y, z]).
/// v' = q * v * conjugate(q). Assumes q is (near) unit.
[Differentiable]
float3 rotate_vector(float3 v, float4 q) {
    float3 u = q.yzw;
    float w = q.x;
    float3 t = 2.0f * cross(u, v);
    return v + w * t + cross(u, t);
}

// =============================================================================
// Ray-Ellipsoid Intersection
// =============================================================================
//
// Stable intersection using closest-approach method instead of
// direct quadratic formula (avoids b² - ac catastrophic cancellation)
// =============================================================================

/// Intersect a ray with an arbitrarily transformed ellipsoid.
/// Returns (t_entry, t_exit) sorted along the ray, or (-1, -1) if no hit.
[Differentiable]
float2 ray_intersect_ellipsoid(
    in float3 ray_origin,
    in float3 ray_dir,    // unit vector
    in float3 center,
    in float3 scale,
    in float4 quat,       // local → world rotation
    in float eps = 1e-6f) // numerical tolerance
{
    // Normalize quaternion for stability
    float4 q = l2_normalize(quat);
    float4 qinv = quat_conjugate(q);

    // Transform ray into ellipsoid local space
    float3 ro = rotate_vector(ray_origin - center, qinv);
    float3 rd = rotate_vector(ray_dir, qinv);

    // Prevent degenerate radii (instant-ngp pattern: use max with small epsilon)
    float3 s = max(abs(scale), eps);
    float3 inv_s = 1.0f / s;

    // Map ellipsoid to unit sphere
    float3 oc = ro * inv_s;
    float3 d = rd * inv_s;

    // Solve |oc + t*d|² = 1  →  a t² + 2 b t + c = 0
    float a = dot(d, d);
    float b = dot(oc, d);

    // Degenerate direction check (instant-ngp pattern)
    const float eps_a = 1e-12f;
    if (a < eps_a) return float2(-1.0f, -1.0f);

    // Stable intersection using closest-approach distance
    // t0 = -b/a is the parameter at closest approach
    float t0 = -b / a;
    float3 p = oc + t0 * d;
    float d2 = dot(p, p);

    // Reject if closest point is outside unit sphere
    if (d2 > 1.0f + eps) return float2(-1.0f, -1.0f);

    // Half-chord length in parameter space
    // thc = sqrt(1 - d²) gives the chord half-length in unit-sphere space
    // Divide by sqrt(a) to convert to ray parameter space
    float thc = safe_sqrt(max(1.0f - d2, 0.0f));
    float inv_sqrt_a = rsqrt(a);

    float t0_hit = t0 - thc * inv_sqrt_a;
    float t1_hit = t0 + thc * inv_sqrt_a;

    // Sort entry/exit (handle grazing rays)
    return float2(min(t0_hit, t1_hit), max(t0_hit, t1_hit));
}

// =============================================================================
// Gradient Regularization Helpers
// =============================================================================
//
// Pattern from instant-ngp/train_nerf.cuh:
//   const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
//   local_dL_doutput[0] = ... + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0]);
// =============================================================================

// L2 regularization term: penalizes large positive values
[Differentiable]
float l2_regularization(float value, float weight = 1e-4f) {
    return max(0.0f, weight * value);
}

// L1 regularization term: penalizes negative density
[Differentiable]
float l1_regularization_negative(float value, float weight = 1e-4f) {
    return value < 0.0f ? -weight : 0.0f;
}

// Near-distance penalty for density (prevents floaters near camera)
// Pattern: (local_network_output[3] > -10.0f && local_depth < near_distance ? 1e-4f : 0.0f)
[Differentiable]
float near_distance_penalty(float density_logit, float depth, float near_distance, float weight = 1e-4f) {
    return (density_logit > -10.0f && depth < near_distance) ? weight : 0.0f;
}
