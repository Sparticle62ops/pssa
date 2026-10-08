//! WGSL implementations of the recurrent and episodic stages.
//!
//! The ownership rules intentionally mirror `src/cuda/stages.ptx`: one
//! invocation owns a token-local row, and the affine scan uses a 256-row
//! workgroup-local Blelloch scan plus recursive tile summaries. The public
//! stage API still publishes the model tape at the same boundaries as CUDA;
//! the forward SSM buffers remain resident until the memory dispatch consumes
//! them.

use super::{WgpuContext, begin_error_scopes, checked_f32_bytes, finish_error_scopes};

pub(crate) const WGSL_STAGE_KERNELS: &str = r#"
struct ScanUniforms { len: u32, stride: u32, _capacity: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> scan_cfg: ScanUniforms;
@group(0) @binding(1) var<storage, read> scan_in_a: array<f32>;
@group(0) @binding(2) var<storage, read> scan_in_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> scan_out_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> scan_out_b: array<f32>;
@group(0) @binding(5) var<storage, read_write> scan_summary_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> scan_summary_b: array<f32>;
var<workgroup> scan_shared_a: array<f32, 256>;
var<workgroup> scan_shared_b: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn affine_scan_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local = lid.x;
    let channel = wid.x;
    let tile = wid.y;
    let time = tile * 256u + local;
    var a = 1.0;
    var b = 0.0;
    if (channel < scan_cfg.stride && time < scan_cfg.len) {
        let index = time * scan_cfg.stride + channel;
        a = scan_in_a[index];
        b = scan_in_b[index];
    }
    scan_shared_a[local] = a;
    scan_shared_b[local] = b;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if (offset >= 256u) { break; }
        let step = offset * 2u;
        if ((local + 1u) % step == 0u) {
            let left = local - offset;
            let old_a = scan_shared_a[local];
            let old_b = scan_shared_b[local];
            scan_shared_a[local] = old_a * scan_shared_a[left];
            scan_shared_b[local] = old_a * scan_shared_b[left] + old_b;
        }
        workgroupBarrier();
        offset = offset * 2u;
    }

    if (local == 0u && channel < scan_cfg.stride) {
        let summary = tile * scan_cfg.stride + channel;
        scan_summary_a[summary] = scan_shared_a[255u];
        scan_summary_b[summary] = scan_shared_b[255u];
        scan_shared_a[255u] = 1.0;
        scan_shared_b[255u] = 0.0;
    }
    workgroupBarrier();

    var down = 128u;
    loop {
        if (down == 0u) { break; }
        let step = down * 2u;
        if ((local + 1u) % step == 0u) {
            let left = local - down;
            let parent_a = scan_shared_a[local];
            let parent_b = scan_shared_b[local];
            let left_a = scan_shared_a[left];
            let left_b = scan_shared_b[left];
            scan_shared_a[left] = parent_a;
            scan_shared_b[left] = parent_b;
            // The parent prefix is applied before the left subtree map.
            scan_shared_a[local] = left_a * parent_a;
            scan_shared_b[local] = left_a * parent_b + left_b;
        }
        workgroupBarrier();
        down = down / 2u;
    }
    if (channel < scan_cfg.stride && time < scan_cfg.len) {
        let index = time * scan_cfg.stride + channel;
        scan_out_a[index] = scan_shared_a[local];
        scan_out_b[index] = scan_shared_b[local];
    }
}

struct ApplyUniforms { len: u32, stride: u32, capacity: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> apply_cfg: ApplyUniforms;
@group(0) @binding(1) var<storage, read_write> apply_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> apply_b: array<f32>;
@group(0) @binding(3) var<storage, read> apply_prefix_a: array<f32>;
@group(0) @binding(4) var<storage, read> apply_prefix_b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn affine_scan_apply_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = apply_cfg.len * apply_cfg.stride;
    if (flat >= total) { return; }
    let channel = flat % apply_cfg.stride;
    let time = flat / apply_cfg.stride;
    let tile = time / apply_cfg.capacity;
    let prefix = tile * apply_cfg.stride + channel;
    let a = apply_a[flat];
    let b = apply_b[flat];
    let pa = apply_prefix_a[prefix];
    let pb = apply_prefix_b[prefix];
    apply_a[flat] = a * pa;
    apply_b[flat] = a * pb + b;
}

struct SsmUniforms { len: u32, dm: u32, ds: u32, stride: u32, scale: f32, _p0: f32, _p1: f32, _p2: f32 };
@group(0) @binding(0) var<uniform> ssm_cfg: SsmUniforms;
@group(0) @binding(1) var<storage, read> ssm_delta: array<f32>;
@group(0) @binding(2) var<storage, read> ssm_b: array<f32>;
@group(0) @binding(3) var<storage, read> ssm_x: array<f32>;
@group(0) @binding(4) var<storage, read> ssm_rates: array<f32>;
@group(0) @binding(5) var<storage, read_write> ssm_local_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> ssm_local_b: array<f32>;
@group(0) @binding(7) var<storage, read_write> ssm_scan_b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn ssm_prepare_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= ssm_cfg.len * ssm_cfg.stride) { return; }
    let t = flat / ssm_cfg.stride;
    let state = flat % ssm_cfg.stride;
    let i = state / ssm_cfg.ds;
    let j = state % ssm_cfg.ds;
    let delta = ssm_delta[t * ssm_cfg.dm + i];
    let a = exp(delta * ssm_rates[state]);
    let b = delta * ssm_b[t * ssm_cfg.ds + j];
    ssm_local_a[flat] = a;
    ssm_local_b[flat] = b;
    ssm_scan_b[flat] = b * ssm_x[t * ssm_cfg.dm + i];
}

@group(0) @binding(8) var<storage, read> ssm_initial: array<f32>;
@group(0) @binding(9) var<storage, read> ssm_c: array<f32>;
@group(0) @binding(10) var<storage, read> ssm_scan_a_in: array<f32>;
@group(0) @binding(11) var<storage, read> ssm_scan_b_in: array<f32>;
@group(0) @binding(12) var<storage, read_write> ssm_states: array<f32>;
@group(0) @binding(13) var<storage, read_write> ssm_y: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn ssm_materialize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= ssm_cfg.len * ssm_cfg.dm) { return; }
    let t = flat / ssm_cfg.dm;
    let i = flat % ssm_cfg.dm;
    var y = 0.0;
    for (var j = 0u; j < ssm_cfg.ds; j = j + 1u) {
        let state = i * ssm_cfg.ds + j;
        let index = t * ssm_cfg.stride + state;
        let before = ssm_scan_a_in[index] * ssm_initial[state] + ssm_scan_b_in[index];
        let h = ssm_local_a[index] * before + ssm_local_b[index] * ssm_x[t * ssm_cfg.dm + i];
        ssm_states[(t + 1u) * ssm_cfg.stride + state] = h;
        y = y + h * ssm_c[t * ssm_cfg.ds + j];
    }
    ssm_y[flat] = y;
}

@group(0) @binding(14) var<storage, read> ssm_bar_a_in: array<f32>;
@group(0) @binding(15) var<storage, read> ssm_c_in: array<f32>;
@group(0) @binding(16) var<storage, read> ssm_gz_in: array<f32>;
@group(0) @binding(17) var<storage, read> ssm_gy_in: array<f32>;
@group(0) @binding(18) var<storage, read_write> ssm_rev_a: array<f32>;
@group(0) @binding(19) var<storage, read_write> ssm_rev_b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn ssm_backward_maps_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= ssm_cfg.len * ssm_cfg.stride) { return; }
    let u = flat / ssm_cfg.stride;
    let state = flat % ssm_cfg.stride;
    let t = ssm_cfg.len - 1u - u;
    let i = state / ssm_cfg.ds;
    let j = state % ssm_cfg.ds;
    let gy = ssm_gz_in[t * ssm_cfg.dm + i] * ssm_cfg.scale + ssm_gy_in[t * ssm_cfg.dm + i];
    let a = ssm_bar_a_in[t * ssm_cfg.stride + state];
    ssm_rev_a[flat] = a;
    ssm_rev_b[flat] = a * gy * ssm_c_in[t * ssm_cfg.ds + j];
}

@group(0) @binding(20) var<storage, read> ssm_raw_in: array<f32>;
@group(0) @binding(21) var<storage, read> ssm_delta_in: array<f32>;
@group(0) @binding(22) var<storage, read> ssm_bproj_in: array<f32>;
@group(0) @binding(23) var<storage, read> ssm_cproj_in: array<f32>;
@group(0) @binding(24) var<storage, read> ssm_rate_in: array<f32>;
@group(0) @binding(25) var<storage, read> ssm_deriv_in: array<f32>;
@group(0) @binding(26) var<storage, read> ssm_xnorm_in: array<f32>;
@group(0) @binding(27) var<storage, read> ssm_states_in: array<f32>;
@group(0) @binding(28) var<storage, read> ssm_bara_local: array<f32>;
@group(0) @binding(29) var<storage, read> ssm_barb_local: array<f32>;
@group(0) @binding(30) var<storage, read> ssm_future_in: array<f32>;
@group(0) @binding(31) var<storage, read> ssm_gz_local: array<f32>;
@group(0) @binding(32) var<storage, read> ssm_gy_local: array<f32>;
@group(0) @binding(33) var<storage, read_write> ssm_gdelta_out: array<f32>;
@group(0) @binding(34) var<storage, read_write> ssm_gb_out: array<f32>;
@group(0) @binding(35) var<storage, read_write> ssm_gc_out: array<f32>;
@group(0) @binding(36) var<storage, read_write> ssm_ga_out: array<f32>;
@group(0) @binding(37) var<storage, read_write> ssm_gx_out: array<f32>;
fn wgsl_sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@compute @workgroup_size(64, 1, 1)
fn ssm_backward_local_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= ssm_cfg.len) { return; }
    let row_d = t * ssm_cfg.dm;
    let row_s = t * ssm_cfg.ds;
    let row_state = t * ssm_cfg.stride;
    let future_u = ssm_cfg.len - 1u - t;
    for (var j = 0u; j < ssm_cfg.ds; j = j + 1u) {
        ssm_gc_out[row_s + j] = 0.0;
        ssm_gb_out[row_s + j] = 0.0;
    }
    for (var i = 0u; i < ssm_cfg.dm; i = i + 1u) {
        let d_off = row_d + i;
        let delta = ssm_delta_in[d_off];
        let x = ssm_xnorm_in[d_off];
        let gy = ssm_gz_local[d_off] * ssm_cfg.scale + ssm_gy_local[d_off];
        var gd = 0.0;
        var gx = 0.0;
        for (var j = 0u; j < ssm_cfg.ds; j = j + 1u) {
            let state = i * ssm_cfg.ds + j;
            let local = row_state + state;
            let future = future_u * ssm_cfg.stride + state;
            let q = gy * ssm_cproj_in[row_s + j] + ssm_future_in[future];
            let h_prev = ssm_states_in[t * ssm_cfg.stride + state];
            let h_next = ssm_states_in[(t + 1u) * ssm_cfg.stride + state];
            let a = ssm_bara_local[local];
            let b = ssm_bproj_in[row_s + j];
            ssm_gc_out[row_s + j] = ssm_gc_out[row_s + j] + gy * h_next;
            ssm_ga_out[local] = q * (delta * a) * h_prev * ssm_deriv_in[state];
            gd = gd + q * (ssm_rate_in[state] * a * h_prev + b * x);
            ssm_gb_out[row_s + j] = ssm_gb_out[row_s + j] + q * (delta * x);
            gx = gx + q * ssm_barb_local[local];
        }
        ssm_gdelta_out[d_off] = gd * wgsl_sigmoid(ssm_raw_in[d_off]);
        ssm_gx_out[d_off] = gx;
    }
}

struct MemoryUniforms { len: u32, dm: u32, dk: u32, dv: u32, cap: u32, count: u32, tau: f32, _pad: f32 };
@group(0) @binding(0) var<uniform> mem_cfg: MemoryUniforms;
@group(0) @binding(1) var<storage, read> mem_x: array<f32>;
@group(0) @binding(2) var<storage, read> mem_y: array<f32>;
@group(0) @binding(3) var<storage, read> mem_wqx: array<f32>;
@group(0) @binding(4) var<storage, read> mem_wqh: array<f32>;
@group(0) @binding(5) var<storage, read> mem_wgate: array<f32>;
@group(0) @binding(6) var<storage, read> mem_wproj: array<f32>;
@group(0) @binding(7) var<storage, read> mem_keys: array<f32>;
@group(0) @binding(8) var<storage, read> mem_norms: array<f32>;
@group(0) @binding(9) var<storage, read> mem_values: array<f32>;
@group(0) @binding(10) var<storage, read_write> mem_qe: array<f32>;
@group(0) @binding(11) var<storage, read_write> mem_qp: array<f32>;
@group(0) @binding(12) var<storage, read_write> mem_qn: array<f32>;
@group(0) @binding(13) var<storage, read_write> mem_weights: array<f32>;
@group(0) @binding(14) var<storage, read_write> mem_mv: array<f32>;
@group(0) @binding(15) var<storage, read_write> mem_gate: array<f32>;
@group(0) @binding(16) var<storage, read_write> mem_mp: array<f32>;
@group(0) @binding(17) var<storage, read_write> mem_inj: array<f32>;
fn memory_distance(q: vec2<f32>, k: vec2<f32>, qsq: f32, ksq: f32) -> f32 {
    let dx = q.x - k.x;
    let dy = q.y - k.y;
    let denom = (1.0 - qsq) * (1.0 - ksq);
    let z = (dx * dx + dy * dy) / denom;
    return 2.0 * log(sqrt(z) + sqrt(1.0 + z));
}
@compute @workgroup_size(64, 1, 1)
fn memory_forward_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= mem_cfg.len) { return; }
    let xoff = t * mem_cfg.dm;
    let qoff = t * mem_cfg.dk;
    let voff = t * mem_cfg.dv;
    for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
        var q = 0.0;
        for (var j = 0u; j < mem_cfg.dm; j = j + 1u) {
            q = q + mem_wqx[r * mem_cfg.dm + j] * mem_x[xoff + j] +
                mem_wqh[r * mem_cfg.dm + j] * mem_y[xoff + j];
        }
        mem_qe[qoff + r] = q;
    }
    // A scaled norm avoids overflowing the f32 accumulator for valid finite
    // host inputs. Normal training values take the same f32 path as the CPU
    // reference to within the documented parity tolerance.
    var max_abs = 0.0;
    for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
        max_abs = max(max_abs, abs(mem_qe[qoff + r]));
    }
    var scaled_sq = 0.0;
    if (max_abs != 0.0) {
        for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
            let q = mem_qe[qoff + r] / max_abs;
            scaled_sq = scaled_sq + q * q;
        }
    }
    let scaled_radius = sqrt(scaled_sq);
    let radius = min(max_abs * scaled_radius, 3.402823e+38);
    let max_radius = 1.0 - 8.0 * 1.1920929e-7;
    var project_scale = 1.0;
    if (max_abs != 0.0) {
        if (radius > 2.0e+6) {
            project_scale = max_radius * (1.0 / max_abs) / scaled_radius;
        } else {
            project_scale = min(radius / (1.0 + radius), max_radius) / radius;
        }
    }
    mem_qn[t] = min(radius, 3.402823e+38);
    var projected_sq = 0.0;
    for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
        let q = mem_qe[qoff + r] * project_scale;
        mem_qp[qoff + r] = q;
        projected_sq = projected_sq + q * q;
    }
    for (var e = 0u; e < mem_cfg.cap; e = e + 1u) {
        mem_weights[t * mem_cfg.cap + e] = 0.0;
    }
    for (var j = 0u; j < mem_cfg.dv; j = j + 1u) {
        mem_mv[voff + j] = 0.0;
    }
    if (mem_cfg.count != 0u) {
        var min_dist = 3.402823e+38;
        for (var e = 0u; e < mem_cfg.count; e = e + 1u) {
            var sq = 0.0;
            for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
                let d = mem_qp[qoff + r] - mem_keys[e * mem_cfg.dk + r];
                sq = sq + d * d;
            }
            let denom = (1.0 - projected_sq) * (1.0 - mem_norms[e]);
            let z = sq / denom;
            let dist = 2.0 * log(sqrt(z) + sqrt(1.0 + z));
            mem_weights[t * mem_cfg.cap + e] = dist;
            min_dist = min(min_dist, dist);
        }
        var sum = 0.0;
        for (var e = 0u; e < mem_cfg.count; e = e + 1u) {
            let idx = t * mem_cfg.cap + e;
            let weight = exp((min_dist - mem_weights[idx]) / mem_cfg.tau);
            mem_weights[idx] = weight;
            sum = sum + weight;
        }
        for (var e = 0u; e < mem_cfg.count; e = e + 1u) {
            let idx = t * mem_cfg.cap + e;
            let weight = mem_weights[idx] / sum;
            mem_weights[idx] = weight;
            for (var j = 0u; j < mem_cfg.dv; j = j + 1u) {
                mem_mv[voff + j] = mem_mv[voff + j] + weight * mem_values[e * mem_cfg.dv + j];
            }
        }
    }
    for (var i = 0u; i < mem_cfg.dm; i = i + 1u) {
        var gate = 0.0;
        var proj = 0.0;
        for (var j = 0u; j < mem_cfg.dm; j = j + 1u) {
            gate = gate + mem_wgate[i * mem_cfg.dm + j] * mem_x[xoff + j];
        }
        for (var j = 0u; j < mem_cfg.dv; j = j + 1u) {
            proj = proj + mem_wproj[i * mem_cfg.dv + j] * mem_mv[voff + j];
        }
        gate = 1.0 / (1.0 + exp(-gate));
        mem_gate[xoff + i] = gate;
        mem_mp[xoff + i] = proj;
        mem_inj[xoff + i] = gate * proj;
    }
}

@group(0) @binding(18) var<storage, read> mem_bwd_gz: array<f32>;
@group(0) @binding(19) var<storage, read> mem_bwd_gate: array<f32>;
@group(0) @binding(20) var<storage, read> mem_bwd_proj: array<f32>;
@group(0) @binding(21) var<storage, read_write> mem_bwd_mproj: array<f32>;
@group(0) @binding(22) var<storage, read_write> mem_bwd_ggate: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn memory_backward_local_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= mem_cfg.len * mem_cfg.dm) { return; }
    let z = mem_bwd_gz[flat]; let gate = mem_bwd_gate[flat];
    mem_bwd_mproj[flat] = z * gate;
    mem_bwd_ggate[flat] = z * mem_bwd_proj[flat] * gate * (1.0 - gate);
}

@group(0) @binding(23) var<storage, read> ret_qp: array<f32>;
@group(0) @binding(24) var<storage, read> ret_qe: array<f32>;
@group(0) @binding(25) var<storage, read> ret_gm: array<f32>;
@group(0) @binding(26) var<storage, read> ret_mv: array<f32>;
@group(0) @binding(27) var<storage, read> ret_weights: array<f32>;
@group(0) @binding(28) var<storage, read> ret_keys: array<f32>;
@group(0) @binding(29) var<storage, read> ret_norms: array<f32>;
@group(0) @binding(30) var<storage, read> ret_values: array<f32>;
@group(0) @binding(31) var<storage, read_write> ret_qp_grad: array<f32>;
@group(0) @binding(32) var<storage, read_write> ret_qe_grad: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn memory_backward_retrieval_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= mem_cfg.len) { return; }
    let qoff = t * mem_cfg.dk;
    var qsq = 0.0;
    for (var r = 0u; r < mem_cfg.dk; r = r + 1u) { let q = ret_qe[qoff + r]; qsq = qsq + q * q; }
    var qpsq = 0.0;
    for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
        let q = ret_qp[qoff + r];
        qpsq = qpsq + q * q;
    }
    for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
        var gp = 0.0;
        for (var e = 0u; e < mem_cfg.count; e = e + 1u) {
            var dot = 0.0;
            for (var j = 0u; j < mem_cfg.dv; j = j + 1u) {
                dot = dot + ret_gm[t * mem_cfg.dv + j] *
                    (ret_values[e * mem_cfg.dv + j] - ret_mv[t * mem_cfg.dv + j]);
            }
            let score = ret_weights[t * mem_cfg.cap + e] * dot;
            var sq = 0.0;
            for (var k = 0u; k < mem_cfg.dk; k = k + 1u) {
                let d = ret_qp[qoff + k] - ret_keys[e * mem_cfg.dk + k];
                sq = sq + d * d;
            }
            if (sq > 0.0) {
                let denom = (1.0 - qpsq) * (1.0 - ret_norms[e]);
                let z = sq / denom;
                let dd_dz = 1.0 / sqrt(z * (1.0 + z));
                let diff = ret_qp[qoff + r] - ret_keys[e * mem_cfg.dk + r];
                let ddenom = -2.0 * ret_qp[qoff + r] * (1.0 - ret_norms[e]);
                let dz = (2.0 * diff * denom - sq * ddenom) / (denom * denom);
                gp = gp + score * (-1.0 / mem_cfg.tau) * dd_dz * dz;
            }
        }
        ret_qp_grad[qoff + r] = gp;
    }
    let radius = sqrt(qsq);
    if (radius == 0.0) {
        for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
            ret_qe_grad[qoff + r] = ret_qp_grad[qoff + r];
        }
    } else {
        let raw_radius = radius / (1.0 + radius);
        let max_radius = 1.0 - 8.0 * 1.1920929e-7;
        let saturated = raw_radius >= max_radius;
        let projected_radius = select(raw_radius, 1.0, saturated);
        let projection_scale = select(1.0 / (1.0 + radius), max_radius / radius, saturated);
        var unit_dot = 0.0;
        for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
            unit_dot = unit_dot + (ret_qe[qoff + r] / radius) * ret_qp_grad[qoff + r];
        }
        for (var r = 0u; r < mem_cfg.dk; r = r + 1u) {
            ret_qe_grad[qoff + r] = projection_scale *
                (ret_qp_grad[qoff + r] - projected_radius *
                (ret_qe[qoff + r] / radius) * unit_dot);
        }
    }
}

@group(0) @binding(33) var<storage, read_write> softplus_values: array<f32>;
@group(0) @binding(34) var<uniform> softplus_len: u32;
@compute @workgroup_size(256, 1, 1)
fn softplus_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= softplus_len) { return; }
    let x = softplus_values[gid.x];
    softplus_values[gid.x] = select(log(1.0 + exp(x)), x, x > 20.0);
}
"#;

const TILE: usize = 256;

#[derive(Default)]
pub(crate) struct WgpuStageState {
    pub(crate) forward: Option<WgpuForwardBuffers>,
}

pub(crate) struct WgpuForwardBuffers {
    pub(crate) len: usize,
    pub(crate) stride: usize,
    pub(crate) bar_a: wgpu::Buffer,
    pub(crate) bar_b: wgpu::Buffer,
    pub(crate) states: wgpu::Buffer,
    pub(crate) y: wgpu::Buffer,
}

fn elems(values: &[usize], label: &str) -> Result<usize, String> {
    let count = values
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| format!("WebGPU {label} element count overflow"))?;
    if count == 0 || count > u32::MAX as usize {
        return Err(format!(
            "WebGPU {label} element count exceeds WGSL indexing"
        ));
    }
    checked_f32_bytes(count)?;
    Ok(count)
}

fn buffer(
    ctx: &WgpuContext,
    data: &[f32],
    elements: usize,
    label: &str,
    copy_src: bool,
) -> Result<wgpu::Buffer, String> {
    let bytes = checked_f32_bytes(elements)?;
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        ctx.queue.write_buffer(&out, 0, bytemuck::cast_slice(data));
    }
    Ok(out)
}

fn uniform(ctx: &WgpuContext, words: &[u8], label: &str) -> wgpu::Buffer {
    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: words.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&out, 0, words);
    out
}

fn dispatch(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry<'_>],
    x: u32,
    y: u32,
    z: u32,
    label: &str,
) -> Result<(), String> {
    let layout = pipeline.get_bind_group_layout(0);
    let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }
    ctx.queue.submit(Some(encoder.finish()));
    finish_error_scopes(&ctx.device)
}

fn read(ctx: &WgpuContext, buf: &wgpu::Buffer, out: &mut [f32]) -> Result<(), String> {
    if out.is_empty() {
        return Ok(());
    }
    let data = ctx.read_buffer_blocking(buf, out.len())?;
    out.copy_from_slice(&data);
    Ok(())
}

fn maps_scan(
    ctx: &WgpuContext,
    input_a: &wgpu::Buffer,
    input_b: &wgpu::Buffer,
    len: usize,
    stride: usize,
) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
    let tiles = len.div_ceil(TILE);
    let total = elems(&[len, stride], "scan")?;
    let summary_len = elems(&[tiles, stride], "scan summaries")?;
    let out_a = buffer(ctx, &[], total, "wgpu_scan_a", true)?;
    let out_b = buffer(ctx, &[], total, "wgpu_scan_b", true)?;
    let summaries_a = buffer(ctx, &[], summary_len, "wgpu_scan_summary_a", true)?;
    let summaries_b = buffer(ctx, &[], summary_len, "wgpu_scan_summary_b", true)?;
    let cfg = uniform(
        ctx,
        bytemuck::cast_slice(&[len as u32, stride as u32, TILE as u32, 0u32]),
        "wgpu_scan_cfg",
    );
    begin_error_scopes(&ctx.device);
    dispatch(
        ctx,
        &ctx.scan_pipeline,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cfg.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: summaries_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: summaries_b.as_entire_binding(),
            },
        ],
        stride as u32,
        tiles as u32,
        1,
        "wgpu_scan",
    )?;
    if tiles > 1 {
        let (prefix_a, prefix_b) = maps_scan(ctx, &summaries_a, &summaries_b, tiles, stride)?;
        let apply_cfg = uniform(
            ctx,
            bytemuck::cast_slice(&[len as u32, stride as u32, TILE as u32, 0u32]),
            "wgpu_scan_apply_cfg",
        );
        begin_error_scopes(&ctx.device);
        dispatch(
            ctx,
            &ctx.scan_apply_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: apply_cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: prefix_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: prefix_b.as_entire_binding(),
                },
            ],
            total.div_ceil(TILE) as u32,
            1,
            1,
            "wgpu_scan_apply",
        )?;
    }
    Ok((out_a, out_b))
}

fn ssm_cfg(
    ctx: &WgpuContext,
    len: usize,
    dm: usize,
    ds: usize,
    scale: f32,
    label: &str,
) -> wgpu::Buffer {
    let values = [len as u32, dm as u32, ds as u32, (dm * ds) as u32];
    let mut bytes = Vec::with_capacity(32);
    bytes.extend_from_slice(bytemuck::cast_slice(&values));
    bytes.extend_from_slice(bytemuck::cast_slice(&[scale, 0.0, 0.0, 0.0]));
    uniform(ctx, &bytes, label)
}

fn memory_cfg(
    ctx: &WgpuContext,
    len: usize,
    dm: usize,
    dk: usize,
    dv: usize,
    cap: usize,
    count: usize,
    tau: f32,
) -> wgpu::Buffer {
    let values = [
        len as u32,
        dm as u32,
        dk as u32,
        dv as u32,
        cap as u32,
        count as u32,
    ];
    let mut bytes = Vec::with_capacity(32);
    bytes.extend_from_slice(bytemuck::cast_slice(&values));
    bytes.extend_from_slice(bytemuck::cast_slice(&[tau, 0.0]));
    uniform(ctx, &bytes, "wgpu_memory_cfg")
}

fn stage_limits(ctx: &WgpuContext, counts: &[usize], label: &str) -> Result<(), String> {
    let limits = ctx.device.limits();
    for &count in counts {
        elems(&[count], label)?;
        let bytes = checked_f32_bytes(count)?;
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
        {
            return Err(format!("WebGPU {label} buffer exceeds device limits"));
        }
    }
    Ok(())
}

impl WgpuContext {
    pub(crate) fn ssm_forward_resident_wgpu(
        &self,
        delta: &[f32],
        delta_raw: &[f32],
        b_proj: &[f32],
        x_norm: &[f32],
        rates: &[f32],
        c_proj: &[f32],
        initial: &[f32],
        len: usize,
        dm: usize,
        ds: usize,
    ) -> Result<(), String> {
        let stride = elems(&[dm, ds], "SSM stride")?;
        let token_state = elems(&[len, stride], "SSM state")?;
        let token_m = elems(&[len, dm], "SSM latent")?;
        let token_s = elems(&[len, ds], "SSM projection")?;
        if delta.len() != token_m
            || delta_raw.len() != token_m
            || x_norm.len() != token_m
            || b_proj.len() != token_s
            || c_proj.len() != token_s
            || rates.len() != stride
            || initial.len() != stride
        {
            return Err("WebGPU SSM forward shape mismatch".into());
        }
        let state_rows = len
            .checked_add(1)
            .ok_or_else(|| "WebGPU SSM state row count overflow".to_string())?;
        let state_len = elems(&[state_rows, stride], "SSM states")?;
        stage_limits(
            self,
            &[token_state, token_m, token_s, stride, state_len],
            "SSM",
        )?;
        let delta_buf = buffer(self, delta, token_m, "wgpu_ssm_delta", false)?;
        let b_buf = buffer(self, b_proj, token_s, "wgpu_ssm_b", false)?;
        let x_buf = buffer(self, x_norm, token_m, "wgpu_ssm_x", false)?;
        let rates_buf = buffer(self, rates, stride, "wgpu_ssm_rates", false)?;
        let c_buf = buffer(self, c_proj, token_s, "wgpu_ssm_c", false)?;
        let initial_buf = buffer(self, initial, stride, "wgpu_ssm_initial", false)?;
        let local_a = buffer(self, &[], token_state, "wgpu_ssm_a", true)?;
        let local_b = buffer(self, &[], token_state, "wgpu_ssm_b_local", true)?;
        let scan_input_b = buffer(self, &[], token_state, "wgpu_ssm_scan_input", true)?;
        let states = buffer(self, &[], state_len, "wgpu_ssm_states", true)?;
        self.queue
            .write_buffer(&states, 0, bytemuck::cast_slice(initial));
        let y = buffer(self, &[], token_m, "wgpu_ssm_y", true)?;
        let cfg = ssm_cfg(self, len, dm, ds, 0.0, "wgpu_ssm_cfg");
        begin_error_scopes(&self.device);
        dispatch(
            self,
            &self.ssm_prepare_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: delta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: rates_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: local_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: local_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: scan_input_b.as_entire_binding(),
                },
            ],
            token_state.div_ceil(256) as u32,
            1,
            1,
            "wgpu_ssm_prepare",
        )?;
        let (scan_a, scan_b) = maps_scan(self, &local_a, &scan_input_b, len, stride)?;
        let material_cfg = ssm_cfg(self, len, dm, ds, 0.0, "wgpu_ssm_material_cfg");
        begin_error_scopes(&self.device);
        dispatch(
            self,
            &self.ssm_materialize_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: local_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: local_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: initial_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: scan_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: scan_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: y.as_entire_binding(),
                },
            ],
            token_m.div_ceil(256) as u32,
            1,
            1,
            "wgpu_ssm_materialize",
        )?;
        let forward = WgpuForwardBuffers {
            len,
            stride,
            bar_a: local_a,
            bar_b: local_b,
            states,
            y,
        };
        self.stages
            .lock()
            .map_err(|_| "WebGPU stage lock poisoned")?
            .forward = Some(forward);
        Ok(())
    }

    pub(crate) fn read_ssm_wgpu(
        &self,
        f: &WgpuForwardBuffers,
        bar_a: &mut [f32],
        bar_b: &mut [f32],
        states: &mut [f32],
        y: &mut [f32],
    ) -> Result<(), String> {
        read(self, &f.bar_a, bar_a)?;
        read(self, &f.bar_b, bar_b)?;
        read(self, &f.states, &mut states[..(f.len + 1) * f.stride])?;
        read(self, &f.y, y)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ssm_forward_wgpu(
        &self,
        delta: &[f32],
        delta_raw: &[f32],
        b_proj: &[f32],
        x_norm: &[f32],
        rates: &[f32],
        _rate_deriv: &[f32],
        c_proj: &[f32],
        initial: &[f32],
        len: usize,
        dm: usize,
        ds: usize,
        bar_a: &mut [f32],
        bar_b: &mut [f32],
        states: &mut [f32],
        y: &mut [f32],
    ) -> Result<(), String> {
        self.ssm_forward_resident_wgpu(
            delta, delta_raw, b_proj, x_norm, rates, c_proj, initial, len, dm, ds,
        )?;
        let mut state = self
            .stages
            .lock()
            .map_err(|_| "WebGPU stage lock poisoned")?;
        let fwd = state.forward.take().ok_or("WebGPU SSM result was lost")?;
        let result = self.read_ssm_wgpu(&fwd, bar_a, bar_b, states, y);
        if result.is_err() {
            state.forward = Some(fwd);
        }
        result
    }

    fn memory_dispatch_wgpu(
        &self,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        wqx: &[f32],
        wqh: &[f32],
        wg: &[f32],
        wp: &[f32],
        keys: &[f32],
        norms: &[f32],
        values: &[f32],
        len: usize,
        dm: usize,
        dk: usize,
        dv: usize,
        cap: usize,
        count: usize,
        tau: f32,
    ) -> Result<[wgpu::Buffer; 8], String> {
        let xlen = elems(&[len, dm], "memory latent")?;
        let qlen = elems(&[len, dk], "memory query")?;
        let wquery = elems(&[dk, dm], "memory query weights")?;
        let wgate = elems(&[dm, dm], "memory gate weights")?;
        let wproj = elems(&[dm, dv], "memory projection weights")?;
        let weights_len = elems(&[len, cap], "memory weights")?;
        let value_len = elems(&[len, dv], "memory values")?;
        let keys_len = elems(&[cap, dk], "memory keys")?;
        let bank_value_len = elems(&[cap, dv], "memory bank values")?;
        if count > cap
            || !tau.is_finite()
            || tau <= 0.0
            || wqx.len() != wquery
            || wqh.len() != wquery
            || wg.len() != wgate
            || wp.len() != wproj
            || keys.len() != keys_len
            || norms.len() != cap
            || values.len() != bank_value_len
        {
            return Err("WebGPU memory forward shape mismatch".into());
        }
        stage_limits(
            self,
            &[xlen, qlen, weights_len, value_len, keys_len, bank_value_len],
            "memory",
        )?;
        let wqx_b = buffer(self, wqx, wquery, "wgpu_mem_wqx", false)?;
        let wqh_b = buffer(self, wqh, wquery, "wgpu_mem_wqh", false)?;
        let wg_b = buffer(self, wg, wgate, "wgpu_mem_gate", false)?;
        let wp_b = buffer(self, wp, wproj, "wgpu_mem_proj", false)?;
        let keys_b = buffer(self, keys, keys_len, "wgpu_mem_keys", false)?;
        let norms_b = buffer(self, norms, cap, "wgpu_mem_norms", false)?;
        let values_b = buffer(self, values, bank_value_len, "wgpu_mem_values", false)?;
        let qe = buffer(self, &[], qlen, "wgpu_mem_qe", true)?;
        let qp = buffer(self, &[], qlen, "wgpu_mem_qp", true)?;
        let qn = buffer(self, &[], len, "wgpu_mem_qn", true)?;
        let weights = buffer(self, &[], weights_len, "wgpu_mem_weights", true)?;
        let mv = buffer(self, &[], value_len, "wgpu_mem_mv", true)?;
        let gate = buffer(self, &[], xlen, "wgpu_mem_gate_out", true)?;
        let mp = buffer(self, &[], xlen, "wgpu_mem_mp", true)?;
        let inj = buffer(self, &[], xlen, "wgpu_mem_inj", true)?;
        let cfg = memory_cfg(self, len, dm, dk, dv, cap, count, tau);
        begin_error_scopes(&self.device);
        dispatch(
            self,
            &self.memory_forward_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wqx_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wqh_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wg_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wp_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: keys_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: norms_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: values_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: qe.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: qp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: qn.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: mv.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: gate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: mp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: inj.as_entire_binding(),
                },
            ],
            len.div_ceil(64) as u32,
            1,
            1,
            "wgpu_memory_forward",
        )?;
        Ok([qe, qp, qn, weights, mv, gate, mp, inj])
    }

    pub(crate) fn memory_forward_wgpu(
        &self,
        x_norm: &[f32],
        y_ssm: &[f32],
        wqx: &[f32],
        wqh: &[f32],
        wg: &[f32],
        wp: &[f32],
        keys: &[f32],
        norms: &[f32],
        values: &[f32],
        len: usize,
        dm: usize,
        dk: usize,
        dv: usize,
        cap: usize,
        count: usize,
        tau: f32,
        qe: &mut [f32],
        qp: &mut [f32],
        qn: &mut [f32],
        weights: &mut [f32],
        mv: &mut [f32],
        gate: &mut [f32],
        mp: &mut [f32],
        inj: &mut [f32],
    ) -> Result<(), String> {
        let xlen = elems(&[len, dm], "memory latent")?;
        let value_len = elems(&[len, dv], "memory values")?;
        let query_len = elems(&[len, dk], "memory query")?;
        let weight_len = elems(&[len, cap], "memory weights")?;
        if x_norm.len() != xlen
            || y_ssm.len() != xlen
            || qe.len() != query_len
            || qp.len() != query_len
            || qn.len() != len
            || weights.len() != weight_len
            || mv.len() != value_len
            || gate.len() != xlen
            || mp.len() != xlen
            || inj.len() != xlen
        {
            return Err("WebGPU memory input/output shape mismatch".into());
        }
        let x = buffer(self, x_norm, xlen, "wgpu_mem_x", false)?;
        let y = buffer(self, y_ssm, xlen, "wgpu_mem_y", false)?;
        let out = self.memory_dispatch_wgpu(
            &x, &y, wqx, wqh, wg, wp, keys, norms, values, len, dm, dk, dv, cap, count, tau,
        )?;
        for (buf, dst) in out.iter().zip([qe, qp, qn, weights, mv, gate, mp, inj]) {
            read(self, buf, dst)?;
        }
        Ok(())
    }

    pub(crate) fn memory_forward_after_ssm_wgpu(
        &self,
        x_norm: &[f32],
        wqx: &[f32],
        wqh: &[f32],
        wg: &[f32],
        wp: &[f32],
        keys: &[f32],
        norms: &[f32],
        values: &[f32],
        len: usize,
        dm: usize,
        dk: usize,
        dv: usize,
        cap: usize,
        count: usize,
        tau: f32,
        bar_a: &mut [f32],
        bar_b: &mut [f32],
        states: &mut [f32],
        y_ssm: &mut [f32],
        qe: &mut [f32],
        qp: &mut [f32],
        qn: &mut [f32],
        weights: &mut [f32],
        mv: &mut [f32],
        gate: &mut [f32],
        mp: &mut [f32],
        inj: &mut [f32],
    ) -> Result<(), String> {
        let mut state = self
            .stages
            .lock()
            .map_err(|_| "WebGPU stage lock poisoned")?;
        let fwd = state
            .forward
            .take()
            .ok_or("WebGPU SSM result is not resident")?;
        drop(state);
        let x = buffer(
            self,
            x_norm,
            elems(&[len, dm], "memory latent")?,
            "wgpu_mem_x_resident",
            false,
        )?;
        let out = self.memory_dispatch_wgpu(
            &x, &fwd.y, wqx, wqh, wg, wp, keys, norms, values, len, dm, dk, dv, cap, count, tau,
        );
        let result = (|| -> Result<(), String> {
            self.read_ssm_wgpu(&fwd, bar_a, bar_b, states, y_ssm)?;
            let out = out?;
            for (buf, dst) in out.iter().zip([qe, qp, qn, weights, mv, gate, mp, inj]) {
                read(self, buf, dst)?;
            }
            Ok(())
        })();
        self.stages
            .lock()
            .map_err(|_| "WebGPU stage lock poisoned")?
            .forward = Some(fwd);
        result
    }

    pub(crate) fn ssm_backward_wgpu(
        &self,
        delta: &[f32],
        raw: &[f32],
        b: &[f32],
        c: &[f32],
        rates: &[f32],
        deriv: &[f32],
        x: &[f32],
        states: &[f32],
        bar_a: &[f32],
        bar_b: &[f32],
        gz: &[f32],
        gy: &[f32],
        len: usize,
        dm: usize,
        ds: usize,
        scale: f32,
        gd: &mut [f32],
        gb: &mut [f32],
        gc: &mut [f32],
        ga: &mut [f32],
        gx: &mut [f32],
    ) -> Result<(), String> {
        let stride = elems(&[dm, ds], "SSM stride")?;
        let token_state = elems(&[len, stride], "SSM backward state")?;
        let token_m = elems(&[len, dm], "SSM backward latent")?;
        let token_s = elems(&[len, ds], "SSM backward projection")?;
        if delta.len() != token_m
            || raw.len() != token_m
            || x.len() != token_m
            || gz.len() != token_m
            || gy.len() != token_m
            || b.len() != token_s
            || c.len() != token_s
            || rates.len() != stride
            || deriv.len() != stride
            || bar_a.len() != token_state
            || bar_b.len() != token_state
            || states.len() < (len + 1) * stride
            || gd.len() != token_m
            || gb.len() != token_s
            || gc.len() != token_s
            || ga.len() != token_state
            || gx.len() != token_m
        {
            return Err("WebGPU SSM backward shape mismatch".into());
        }
        let delta_b = buffer(self, delta, token_m, "wgpu_bwd_delta", false)?;
        let raw_b = buffer(self, raw, token_m, "wgpu_bwd_raw", false)?;
        let b_b = buffer(self, b, token_s, "wgpu_bwd_b", false)?;
        let c_b = buffer(self, c, token_s, "wgpu_bwd_c", false)?;
        let rates_b = buffer(self, rates, stride, "wgpu_bwd_rates", false)?;
        let deriv_b = buffer(self, deriv, stride, "wgpu_bwd_deriv", false)?;
        let x_b = buffer(self, x, token_m, "wgpu_bwd_x", false)?;
        let states_b = buffer(self, states, (len + 1) * stride, "wgpu_bwd_states", false)?;
        let bara_b = buffer(self, bar_a, token_state, "wgpu_bwd_a", false)?;
        let barb_b = buffer(self, bar_b, token_state, "wgpu_bwd_bara", false)?;
        let gz_b = buffer(self, gz, token_m, "wgpu_bwd_gz", false)?;
        let gy_b = buffer(self, gy, token_m, "wgpu_bwd_gy", false)?;
        let rev_a = buffer(self, &[], token_state, "wgpu_bwd_rev_a", true)?;
        let rev_b = buffer(self, &[], token_state, "wgpu_bwd_rev_b", true)?;
        let cfg = ssm_cfg(self, len, dm, ds, scale, "wgpu_bwd_cfg");
        begin_error_scopes(&self.device);
        dispatch(
            self,
            &self.ssm_backward_maps_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: bara_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: c_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: gz_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: gy_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: rev_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: rev_b.as_entire_binding(),
                },
            ],
            token_state.div_ceil(256) as u32,
            1,
            1,
            "wgpu_bwd_maps",
        )?;
        let (_scan_a, scan_b) = maps_scan(self, &rev_a, &rev_b, len, stride)?;
        let out_gd = buffer(self, &[], token_m, "wgpu_bwd_gd", true)?;
        let out_gb = buffer(self, &[], token_s, "wgpu_bwd_gb", true)?;
        let out_gc = buffer(self, &[], token_s, "wgpu_bwd_gc", true)?;
        let out_ga = buffer(self, &[], token_state, "wgpu_bwd_ga", true)?;
        let out_gx = buffer(self, &[], token_m, "wgpu_bwd_gx", true)?;
        begin_error_scopes(&self.device);
        dispatch(
            self,
            &self.ssm_backward_local_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: raw_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: delta_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: b_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: c_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: rates_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: deriv_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: x_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 27,
                    resource: states_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 28,
                    resource: bara_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 29,
                    resource: barb_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 30,
                    resource: scan_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 31,
                    resource: gz_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 32,
                    resource: gy_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 33,
                    resource: out_gd.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 34,
                    resource: out_gb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 35,
                    resource: out_gc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 36,
                    resource: out_ga.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 37,
                    resource: out_gx.as_entire_binding(),
                },
            ],
            len.div_ceil(64) as u32,
            1,
            1,
            "wgpu_bwd_local",
        )?;
        read(self, &out_gd, gd)?;
        read(self, &out_gb, gb)?;
        read(self, &out_gc, gc)?;
        read(self, &out_ga, ga)?;
        read(self, &out_gx, gx)
    }

    pub(crate) fn memory_backward_local_wgpu(
        &self,
        gz: &[f32],
        gate: &[f32],
        proj: &[f32],
        out_m: &mut [f32],
        out_g: &mut [f32],
    ) -> Result<(), String> {
        if gz.len() != gate.len()
            || gz.len() != proj.len()
            || gz.len() != out_m.len()
            || gz.len() != out_g.len()
        {
            return Err("WebGPU memory backward local shape mismatch".into());
        }
        let len = gz.len();
        let dm = 1usize;
        let gz_b = buffer(self, gz, len, "wgpu_mem_bwd_gz", false)?;
        let gate_b = buffer(self, gate, len, "wgpu_mem_bwd_gate", false)?;
        let proj_b = buffer(self, proj, len, "wgpu_mem_bwd_proj", false)?;
        let om = buffer(self, &[], len, "wgpu_mem_bwd_m", true)?;
        let og = buffer(self, &[], len, "wgpu_mem_bwd_g", true)?;
        let cfg = memory_cfg(self, len, dm, 1, 1, 1, 0, 1.0);
        begin_error_scopes(&self.device);
        dispatch(
            self,
            &self.memory_backward_local_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: gz_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: gate_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: proj_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: om.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: og.as_entire_binding(),
                },
            ],
            len.div_ceil(64) as u32,
            1,
            1,
            "wgpu_mem_bwd_local",
        )?;
        read(self, &om, out_m)?;
        read(self, &og, out_g)
    }

    pub(crate) fn memory_backward_retrieval_wgpu(
        &self,
        qp: &[f32],
        qe: &[f32],
        gm: &[f32],
        mv: &[f32],
        weights: &[f32],
        keys: &[f32],
        norms: &[f32],
        values: &[f32],
        len: usize,
        count: usize,
        cap: usize,
        dk: usize,
        dv: usize,
        tau: f32,
        out_qp: &mut [f32],
        out_qe: &mut [f32],
    ) -> Result<(), String> {
        let qlen = elems(&[len, dk], "memory backward query")?;
        let weights_len = elems(&[len, cap], "memory backward weights")?;
        let keys_len = elems(&[cap, dk], "memory backward keys")?;
        let values_len = elems(&[cap, dv], "memory backward values")?;
        let value_rows = elems(&[len, dv], "memory backward values")?;
        if count > cap
            || !tau.is_finite()
            || tau <= 0.0
            || qp.len() != qlen
            || qe.len() != qlen
            || out_qp.len() != qlen
            || out_qe.len() != qlen
            || gm.len() != value_rows
            || mv.len() != value_rows
            || weights.len() != weights_len
            || keys.len() != keys_len
            || norms.len() != cap
            || values.len() != values_len
        {
            return Err("WebGPU memory backward retrieval shape mismatch".into());
        }
        let cfg = memory_cfg(self, len, 1, dk, dv, cap, count, tau);
        let b = |v: &[f32], n: &str| buffer(self, v, v.len(), n, false);
        let qp_b = b(qp, "wgpu_ret_qp")?;
        let qe_b = b(qe, "wgpu_ret_qe")?;
        let gm_b = b(gm, "wgpu_ret_gm")?;
        let mv_b = b(mv, "wgpu_ret_mv")?;
        let wt_b = b(weights, "wgpu_ret_w")?;
        let keys_b = b(keys, "wgpu_ret_keys")?;
        let norms_b = b(norms, "wgpu_ret_norms")?;
        let values_b = b(values, "wgpu_ret_values")?;
        let oqp = buffer(self, &[], qlen, "wgpu_ret_out_p", true)?;
        let oqe = buffer(self, &[], qlen, "wgpu_ret_out_e", true)?;
        begin_error_scopes(&self.device);
        dispatch(
            self,
            &self.memory_backward_retrieval_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: qp_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: qe_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: gm_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: mv_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 27,
                    resource: wt_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 28,
                    resource: keys_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 29,
                    resource: norms_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 30,
                    resource: values_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 31,
                    resource: oqp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 32,
                    resource: oqe.as_entire_binding(),
                },
            ],
            len.div_ceil(64) as u32,
            1,
            1,
            "wgpu_ret",
        )?;
        read(self, &oqp, out_qp)?;
        read(self, &oqe, out_qe)
    }

    pub(crate) fn softplus_wgpu(&self, values: &mut [f32]) -> Result<(), String> {
        if values.is_empty() {
            return Ok(());
        }
        let input = buffer(self, values, values.len(), "wgpu_softplus", true)?;
        let len = uniform(
            self,
            bytemuck::cast_slice(&[values.len() as u32]),
            "wgpu_softplus_len",
        );
        begin_error_scopes(&self.device);
        dispatch(
            self,
            &self.softplus_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 33,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 34,
                    resource: len.as_entire_binding(),
                },
            ],
            values.len().div_ceil(256) as u32,
            1,
            1,
            "wgpu_softplus",
        )?;
        read(self, &input, values)
    }
}
