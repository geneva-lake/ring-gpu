use std::error::Error;
use std::os::unix::io::RawFd;
use std::thread;

use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferSubmitInfo,
    CommandBufferUsage, CopyBufferInfo, SemaphoreSubmitInfo, SubmitInfo,
};
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{Device, DeviceCreateInfo, DeviceFeatures, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::library::VulkanLibrary;
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayout};
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::sync::semaphore::{
    Semaphore, SemaphoreCreateInfo, SemaphoreType, SemaphoreWaitInfo,
};
use io_uring::{opcode, types, IoUring};

const WORKGROUP_SIZE: u32 = 256;

#[derive(Clone)]
pub struct GpuContext {
    device: std::sync::Arc<Device>,
    queue: std::sync::Arc<vulkano::device::Queue>,
    memory_allocator: std::sync::Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: std::sync::Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: std::sync::Arc<StandardCommandBufferAllocator>,
    pipeline_map: std::sync::Arc<ComputePipeline>,
    pipeline_reduce: std::sync::Arc<ComputePipeline>,
}

pub fn init_gpu() -> Result<std::sync::Arc<GpuContext>, Box<dyn Error>> {
    let library = VulkanLibrary::new()?;
    let instance = Instance::new(library, InstanceCreateInfo::default())?;
    let physical = instance
        .enumerate_physical_devices()?
        .next()
        .ok_or("No Vulkan physical device found")?;

    let queue_family_index = physical
        .queue_family_properties()
        .iter()
        .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
        .ok_or("No compute queue family found")? as u32;

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_features: DeviceFeatures {
                timeline_semaphore: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    let queue = queues.next().ok_or("No queue created")?;

    let memory_allocator =
        std::sync::Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator =
        std::sync::Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
    let command_buffer_allocator =
        std::sync::Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

    let shader_source = load_shader_source()?;
    let spirv = compile_wgsl_to_spirv(&shader_source)?;
    let shader = unsafe {
        ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&spirv))?
    };

    let stage_map = PipelineShaderStageCreateInfo::new(shader.entry_point("map_reduce").unwrap());
    let layout_map_info =
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_map])
            .into_pipeline_layout_create_info(device.clone())?;
    let layout_map = PipelineLayout::new(device.clone(), layout_map_info)?;
    let pipeline_map = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage_map, layout_map),
    )?;

    let stage_reduce =
        PipelineShaderStageCreateInfo::new(shader.entry_point("final_reduce").unwrap());
    let layout_reduce_info =
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_reduce])
            .into_pipeline_layout_create_info(device.clone())?;
    let layout_reduce = PipelineLayout::new(device.clone(), layout_reduce_info)?;
    let pipeline_reduce = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage_reduce, layout_reduce),
    )?;

    Ok(std::sync::Arc::new(GpuContext {
        device,
        queue,
        memory_allocator,
        descriptor_set_allocator,
        command_buffer_allocator,
        pipeline_map,
        pipeline_reduce,
    }))
}

pub fn run_gpu(
    gpu: std::sync::Arc<GpuContext>,
    data_a: Box<[f32]>,
    data_b: Box<[f32]>,
    main_ring_fd: RawFd,
) -> Result<(), Box<dyn Error>> {

    let buf_a = Buffer::from_iter(
        gpu.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_a.iter().cloned(),
    )?;

    let buf_b = Buffer::from_iter(
        gpu.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_b.iter().cloned(),
    )?;

    let mut current_size = data_a.len();

    let ping = make_zero_buffer(&gpu.memory_allocator, current_size)?;
    let pong = make_zero_buffer(&gpu.memory_allocator, current_size)?;

    let staging = Buffer::from_iter(
        gpu.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        [0.0f32],
    )?;

    let set_map = DescriptorSet::new(
        gpu.descriptor_set_allocator.clone(),
        gpu.pipeline_map.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buf_a.clone()),
            WriteDescriptorSet::buffer(1, buf_b.clone()),
            WriteDescriptorSet::buffer(2, ping.clone()),
        ],
        [],
    )?;

    let set_reduce_1 = DescriptorSet::new(
        gpu.descriptor_set_allocator.clone(),
        gpu.pipeline_reduce.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, ping.clone()),
            WriteDescriptorSet::buffer(2, pong.clone()),
        ],
        [],
    )?;

    let set_reduce_2 = DescriptorSet::new(
        gpu.descriptor_set_allocator.clone(),
        gpu.pipeline_reduce.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, pong.clone()),
            WriteDescriptorSet::buffer(2, ping.clone()),
        ],
        [],
    )?;

    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    let mut dispatch_count =
        (current_size + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize;

    unsafe {
        builder
            .bind_pipeline_compute(gpu.pipeline_map.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                gpu.pipeline_map.layout().clone(),
                0,
                set_map,
            )?
            .dispatch([dispatch_count as u32, 1, 1])?;
    }

    let mut output_is_ping = true;
    current_size = dispatch_count;

    while current_size > 1 {
        dispatch_count =
            (current_size + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize;

        unsafe {
            builder
                .bind_pipeline_compute(gpu.pipeline_reduce.clone())?
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    gpu.pipeline_reduce.layout().clone(),
                    0,
                    if output_is_ping {
                        set_reduce_1.clone()
                    } else {
                        set_reduce_2.clone()
                    },
                )?
                .dispatch([dispatch_count as u32, 1, 1])?;
        }

        current_size = dispatch_count;
        output_is_ping = !output_is_ping;
    }

    let final_buf = if output_is_ping { ping.clone() } else { pong.clone() };
    builder.copy_buffer(CopyBufferInfo::buffers(final_buf, staging.clone()))?;

    let semaphore = std::sync::Arc::new(Semaphore::new(
        gpu.device.clone(),
        SemaphoreCreateInfo {
            semaphore_type: SemaphoreType::Timeline,
            initial_value: 0,
            ..Default::default()
        },
    )?);

    let command_buffer = builder.build()?;
    let mut signal_info = SemaphoreSubmitInfo::new(semaphore.clone());
    signal_info.value = 1;

    gpu.queue.with(|mut q| unsafe {
        q.submit_unchecked(
            &[SubmitInfo {
                command_buffers: vec![CommandBufferSubmitInfo::new(command_buffer)],
                signal_semaphores: vec![signal_info],
                ..Default::default()
            }],
            None,
        )
    })?;

    let keep_alive = (buf_a, buf_b, ping, pong);

    // Wait gpu semafore, then notify the main ring.
    thread::spawn(move || {
        let _keep_alive = keep_alive;

        if semaphore
            .wait(
                SemaphoreWaitInfo {
                    value: 1,
                    ..Default::default()
                },
                None,
            )
            .is_err()
        {
            eprintln!("GPU semaphore wait failed");
            return;
        }

        let result = match staging.read() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("GPU staging read failed: {:?}", e);
                return;
            }
        };
        let value = result[0];

        let ptr = Box::into_raw(Box::new(value)) as u64;

        let mut msg_ring = match IoUring::new(1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("io_uring init failed: {:?}", e);
                return;
            }
        };
        let msg_e = opcode::MsgRingData::new(
            types::Fd(main_ring_fd),
            0,
            ptr,
            None,
        )
        .build();

        unsafe {
            if msg_ring.submission().push(&msg_e).is_err() {
                eprintln!("io_uring submission push failed");
                return;
            }
        }
        if let Err(e) = msg_ring.submit_and_wait(1) {
            eprintln!("io_uring submit failed: {:?}", e);
        }
    });

    Ok(())

}

fn make_zero_buffer(
    allocator: &std::sync::Arc<StandardMemoryAllocator>,
    len: usize,
) -> Result<Subbuffer<[f32]>, Box<dyn Error>> {
    let zeros = std::iter::repeat(0.0f32).take(len);
    let buffer = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            // Buffer::from_iter writes via host mapping, so we need host-visible memory.
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        zeros,
    )?;
    Ok(buffer)
}

fn compile_wgsl_to_spirv(src: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let module = wgsl::parse_str(src)?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator.validate(&module)?;
    let spv = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        None,
    )?;
    Ok(spv)
}

fn load_shader_source() -> Result<String, Box<dyn Error>> {
    let path = format!("{}/src/shader.wgsl", env!("CARGO_MANIFEST_DIR"));
    Ok(std::fs::read_to_string(path)?)
}
