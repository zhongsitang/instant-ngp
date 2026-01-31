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
// Volume Integrator for Ellipsoid Primitive Ray Tracing
// =============================================================================
//
// This module implements volume rendering integration for ellipsoid primitives.
// Each ellipsoid contributes to the integral when a ray passes through it:
// - At entry point: adds +density to the accumulated density field
// - At exit point: adds -density to the accumulated density field
//
// Numerical stability patterns based on instant-ngp implementation:
// - Alpha = 1 - exp(-sigma * dt) using expm1 for small values
// - Weight = alpha * T where T = 1 - accumulated_alpha
// - Early termination when transmittance < MIN_TRANSMITTANCE
// - Clamped exponentials to prevent overflow
// - Safe division with proper epsilon handling
// =============================================================================

import math_utils;

// =============================================================================
// Constants
// =============================================================================

// Minimum accumulated alpha before we stop caring about depth contribution
static const float DEPTH_ALPHA_THRESHOLD = 1e-6f;

// Regularization weight for gradient smoothing
static const float GRADIENT_L2_REG = 1e-4f;

// =============================================================================
// Core Data Structures
// =============================================================================

/// Volume rendering state per ray (48 bytes, 16-byte aligned)
///
/// Memory layout must match C++ IntegratorState in types.h
/// Using float4 for C to avoid float3 alignment issues between C++ and Slang
struct IntegratorState : IDifferentiable
{
    float4 contrib;              // density, r*d, g*d, b*d (current density field)
    float4 C;                    // accumulated color RGB (w unused, for alignment)
    float logT;                  // log(T) = -tau_total, always <= 0
    float depth_num;             // depth numerator (divide by alpha for expected depth)
    float t;                     // current t parameter along ray
    float _pad;                  // padding to 48 bytes
};

/// Sample point along a ray, representing an ellipsoid boundary crossing.
///
/// When a ray crosses an ellipsoid boundary:
/// - Entry (hit_type=1): contrib.x = +density
/// - Exit (hit_type=0): contrib.x = -density
/// The RGB components are density-weighted: contrib.yzw = sign * density * color
struct SamplePoint : IDifferentiable
{
    float t;        // Ray parameter at this sample
    float4 contrib; // (signed_density, r*signed_density, g*signed_density, b*signed_density)
}

// =============================================================================
// Volume Integration
// =============================================================================
//
// Volume rendering equation (discrete form):
//   C = Sum_i T_i * alpha_i * c_i
//   where T_i = exp(-Sum_j tau_j) is transmittance, alpha_i = 1-exp(-tau_i)
//
// Key insight: We process hits in sorted order. Between consecutive hits,
// the density field is constant (sum of overlapping ellipsoids' densities).
//
// Numerical stability (based on instant-ngp):
//   - Use alpha = -expm1(-tau) instead of 1 - exp(-tau) for small tau
//   - Store log(T) instead of T to avoid underflow
//   - Early terminate when T < MIN_TRANSMITTANCE
//   - Clamp all exponential inputs
// =============================================================================

/// Process one volume integration step at a boundary crossing.
///
/// Volume rendering for segment [prev_t, curr_t] with constant density sigma:
///   tau = sigma * (curr_t - prev_t)           // optical depth
///   T_new = T_old * exp(-tau)                 // transmittance update
///   C_new = C_old + T_old * alpha * color     // color accumulation
///
/// Pattern from instant-ngp (render_nerf.cuh:147-150):
///   float alpha = 1.f - __expf(-network_to_density(nerf_out.w, density_activation) * dt);
///   float weight = alpha * (1.0f - color.a);
///   color += vec4(rgb * weight, weight);
[Differentiable]
IntegratorState integrate_step(
    in IntegratorState state,
    in SamplePoint sample)
{
    IntegratorState new_state = state;

    // --- Step 1: Integrate segment [state.t, sample.t] with current density ---

    // Compute segment length (non-negative)
    const float dt = max(sample.t - state.t, 0.0f);

    // Current density (can be sum of overlapping ellipsoids)
    const float sigma = state.contrib.x;

    // Optical depth for this segment
    // tau = sigma * dt, clamped to non-negative (negative density is unphysical)
    const float tau = max(sigma * dt, 0.0f);

    // Extract color from density-weighted contribution
    // color = contrib.yzw / sigma (safe division handles sigma near zero)
    const float3 color = safe_div_with_min(state.contrib.yzw, sigma, TINY_VAL);

    // Update log-transmittance: logT_new = logT_old - tau
    // logT is always <= 0 (transmittance in [0, 1])
    new_state.logT = state.logT - tau;

    // Compute alpha using numerically stable formula
    // alpha = 1 - exp(-tau) = -expm1(-tau)
    // This avoids catastrophic cancellation when tau is small
    const float alpha = tau_to_alpha(tau);

    // Recover transmittance from log-space
    // T = exp(logT), clamped for numerical safety
    const float T = safe_exp(state.logT);

    // Compute compositing weight: weight = alpha * T
    // Pattern from instant-ngp: weight = alpha * (1.0f - color.a)
    // where (1 - color.a) represents accumulated transmittance T
    const float weight = alpha * T;

    // Clamp weight to [0, 1] for safety
    // This prevents numerical issues from affecting accumulated color
    const float weight_clamped = clip(weight, 0.0f, 1.0f);

    // Accumulate color: C_new = C_old + weight * color
    new_state.C.xyz = state.C.xyz + weight_clamped * color;

    // Accumulate depth (expected depth computation)
    // depth_term handles the integral of t * w(t) over the segment
    // For constant density: integral of t * sigma * exp(-sigma*(t-t0)) from t0 to t1
    const float depth_term = compute_depth_term(alpha, tau, sigma);
    new_state.depth_num = state.depth_num + T * (state.t * alpha + depth_term);

    // --- Step 2: Update density field for next segment ---
    // At boundary crossing, density changes by sample.contrib
    new_state.contrib = state.contrib + sample.contrib;
    new_state.t = sample.t;

    return new_state;
}

/// Compute depth integration term for constant density segment.
///
/// For constant density sigma over [0, dt], the depth integral is:
///   integral_0^dt of (t * sigma * exp(-sigma*t)) dt
///   = (alpha - tau * exp(-tau)) / sigma
///
/// This is numerically stable when computed carefully.
[Differentiable]
float compute_depth_term(float alpha, float tau, float sigma) {
    if (abs(sigma) < TINY_VAL) {
        // No density contribution to depth
        return 0.0f;
    }

    // For small tau, use Taylor expansion to avoid cancellation:
    // alpha - tau * exp(-tau) ≈ tau - tau²/2 - tau * (1 - tau + tau²/2)
    //                         = tau - tau²/2 - tau + tau² - tau³/2
    //                         = tau²/2 - tau³/2
    //                         ≈ tau²/2 for small tau
    if (tau < 0.01f) {
        // Taylor approximation: (alpha - tau*exp(-tau))/sigma ≈ dt²*sigma/2
        float dt = safe_div(tau, sigma);
        return 0.5f * dt * dt * sigma;
    }

    // Standard formula for larger tau
    float exp_neg_tau = safe_exp(-tau);
    return safe_div(alpha - tau * exp_neg_tau, sigma);
}

/// Reconstruct previous state from current state (inverse of integrate_step).
/// Used in backward pass to recover intermediate states without storing them.
///
/// Note: This is an approximate inverse due to floating-point precision.
/// For training, consider storing checkpoints instead of relying on inversion.
IntegratorState invert_step(
    in IntegratorState new_state,
    in SamplePoint new_sample,
    in SamplePoint prev_sample)
{
    IntegratorState prev_state = new_state;

    // --- Undo Step 2: Recover previous contrib and t ---
    prev_state.contrib = new_state.contrib - new_sample.contrib;
    prev_state.t = prev_sample.t;

    // --- Undo Step 1: Subtract segment's contribution ---
    const float dt = max(new_sample.t - prev_sample.t, 0.0f);
    const float sigma = prev_state.contrib.x;
    const float tau = max(sigma * dt, 0.0f);
    const float3 color = safe_div_with_min(prev_state.contrib.yzw, sigma, TINY_VAL);

    // Recover log-transmittance: logT_prev = logT_new + tau
    // Clamp to <= 0 (T <= 1)
    prev_state.logT = min(new_state.logT + tau, 0.0f);

    const float alpha = tau_to_alpha(tau);
    const float T = safe_exp(prev_state.logT);
    const float weight = clip(alpha * T, 0.0f, 1.0f);

    // Subtract color contribution
    prev_state.C.xyz = new_state.C.xyz - weight * color;

    // Subtract depth contribution
    const float depth_term = compute_depth_term(alpha, tau, sigma);
    prev_state.depth_num = new_state.depth_num - T * (prev_sample.t * alpha + depth_term);

    return prev_state;
}

// =============================================================================
// Early Termination Check
// =============================================================================

/// Check if ray integration should terminate early.
/// Pattern from instant-ngp (render_nerf.cuh:163):
///   if (color.a > (1.0f - min_transmittance))
///
/// We use logT instead: if logT < log(min_transmittance), terminate
bool should_terminate_integration(in IntegratorState state, float min_transmittance = MIN_TRANSMITTANCE) {
    // T = exp(logT), so T < threshold means logT < log(threshold)
    // log(0.01) ≈ -4.6
    const float log_threshold = safe_log(min_transmittance);
    return state.logT < log_threshold;
}

/// Get current accumulated alpha from state
[Differentiable]
float get_accumulated_alpha(in IntegratorState state) {
    // alpha = 1 - T = 1 - exp(logT) = -expm1(logT)
    return -safe_expm1(state.logT);
}

// =============================================================================
// Output Extraction
// =============================================================================

/// Final rendered output for a ray.
struct RenderOutput : IDifferentiable
{
    float3 C;    // Accumulated color
    float alpha; // Opacity (1 - transmittance)
    float depth; // Expected depth
};

/// Extract final render output from volume integration state.
///
/// Computes:
///   - alpha = 1 - T = -expm1(logT) (opacity from transmittance)
///   - depth = E[t] = depth_num / alpha (expected depth, or 0 if transparent)
///
/// Pattern from instant-ngp (render_nerf.cuh:177):
///   depth_buffer[idx] = color.a > 0.2f ? best_depth_candidate : MAX_DEPTH();
[Differentiable]
RenderOutput extract_output(in IntegratorState state)
{
    // Compute alpha = 1 - exp(logT) using expm1 for stability
    float alpha = -safe_expm1(state.logT);

    // Compute expected depth
    // Only valid when alpha is significantly non-zero
    float depth;
    if (alpha < DEPTH_ALPHA_THRESHOLD) {
        depth = 0.0f;
    } else {
        depth = safe_div(state.depth_num, alpha);
    }

    return { state.C.xyz, alpha, depth };
}

/// Extract output with background compositing
/// Pattern from instant-ngp (train_nerf.cuh:251-253):
///   if (1.0f - color.a >= EPSILON) {
///       color.rgb() += (1.0f - color.a) * background_color;
///   }
[Differentiable]
RenderOutput extract_output_with_background(
    in IntegratorState state,
    in float3 background_color)
{
    RenderOutput output = extract_output(state);

    // Composite background behind accumulated color
    float T = safe_exp(state.logT);  // remaining transmittance
    if (T > ALPHA_EPSILON) {
        output.C = output.C + T * background_color;
    }

    return output;
}

// =============================================================================
// Initialization
// =============================================================================

/// Create initial integrator state for a new ray
IntegratorState init_integrator_state(float t_start = 0.0f)
{
    IntegratorState state;
    state.contrib = float4(0.0f);  // no density initially
    state.C = float4(0.0f);        // no accumulated color
    state.logT = 0.0f;             // T = 1 (full transmittance)
    state.depth_num = 0.0f;
    state.t = t_start;
    state._pad = 0.0f;
    return state;
}

// =============================================================================
// Gradient Regularization (Optional)
// =============================================================================
//
// For training stability, based on instant-ngp patterns:
// - L2 regularization on large color outputs
// - L1 regularization on negative density
// - Near-distance penalty for floaters
// =============================================================================

/// Apply gradient regularization to density gradient
/// Pattern from instant-ngp (train_nerf.cuh:429-430):
///   local_dL_doutput[3] = loss_scale * dloss_by_dmlp +
///       (float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
///       (float(local_network_output[3]) > -10.0f && local_depth < near_distance ? 1e-4f : 0.0f);
[Differentiable]
float regularize_density_gradient(
    float density_gradient,
    float density_logit,
    float depth,
    float near_distance,
    float l1_reg = 0.0f,
    float near_penalty = 1e-4f)
{
    float reg = density_gradient;

    // L1 regularization on negative density (pushes towards non-negative)
    if (density_logit < 0.0f) {
        reg -= l1_reg;
    }

    // Near-distance penalty (prevents floaters near camera)
    if (density_logit > -10.0f && depth < near_distance) {
        reg += near_penalty;
    }

    return reg;
}

/// Apply L2 regularization to color gradients
/// Pattern from instant-ngp (train_nerf.cuh:415-423):
///   fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])
[Differentiable]
float3 regularize_color_gradient(
    float3 color_gradient,
    float3 network_output,
    float l2_reg = GRADIENT_L2_REG)
{
    return color_gradient + float3(
        max(0.0f, l2_reg * network_output.x),
        max(0.0f, l2_reg * network_output.y),
        max(0.0f, l2_reg * network_output.z)
    );
}
