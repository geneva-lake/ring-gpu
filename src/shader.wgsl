@group(0) @binding(0) var<storage, read> inputA: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> inputB: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn map_reduce(
    @builtin(global_invocation_id) g_id: vec3<u32>,
    @builtin(local_invocation_id) l_id: vec3<u32>,
    @builtin(workgroup_id) gr_id: vec3<u32>,
) {
    let i = g_id.x;
    if (i < arrayLength(&inputA)) {
        shared_sums[l_id.x] = dot(inputA[i], inputB[i]);
    } else {
        shared_sums[l_id.x] = 0.0;
    }
    reduce(l_id.x, gr_id.x);
}

@group(0) @binding(0) var<storage, read> inputNext: array<f32>;

@compute @workgroup_size(256)
fn final_reduce(
    @builtin(global_invocation_id) g_id: vec3<u32>,
    @builtin(local_invocation_id) l_id: vec3<u32>,
    @builtin(workgroup_id) gr_id: vec3<u32>,
) {
    if (l_id.x < arrayLength(&inputNext)) {
        shared_sums[l_id.x] = inputNext[g_id.x];
    } else {
        shared_sums[l_id.x] = 0.0;
    }
    reduce(l_id.x, gr_id.x);
}

fn reduce(l_id_x: u32, gr_id_x: u32) {
    workgroupBarrier();
    for (var s = 1u; s < 256u; s <<= 1u) {
        let index = 2u * s * l_id_x;
        if (index < 256u) {
            shared_sums[index] += shared_sums[index + s];
        }
        workgroupBarrier();
    }
    if (l_id_x == 0u) {
        output[gr_id_x] = shared_sums[0];
    }
}
