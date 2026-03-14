use std::error::Error;
use std::os::unix::io::{AsRawFd, RawFd};
use std::thread;
use std::time::Duration;

use io_uring::{opcode, types, IoUring};

use crate::gpu::{self, GpuContext};

const ARRAY_SIZE: usize = 16_777_216;
const TIMEOUT_UD: u64 = 0x5449_4d45_4f55_54; // "TIMEOUT"

pub fn data_flow(
    gpu: std::sync::Arc<GpuContext>,
    ring_fd: RawFd,
) -> Result<(), Box<dyn Error>> {

    let mut msg_ring_listener = IoUring::new(50).unwrap();
    let listener_fd = msg_ring_listener.as_raw_fd();

    // thread receives data from data producer threads and send it to gpu
    thread::spawn(move || {
        loop {
            let timeout = types::Timespec::from(Duration::from_secs(30));
            let timeout_e = opcode::Timeout::new(&timeout)
                .build()
                .user_data(TIMEOUT_UD);
            unsafe {
                msg_ring_listener.submission().push(&timeout_e).ok();
            }

            let mut results: Vec<Box<[f32]>> = Vec::new();
            let _ = msg_ring_listener.submit_and_wait(2);
            let mut cq = msg_ring_listener.completion();
            while let Some(cqe) = cq.next() {
                if cqe.user_data() == TIMEOUT_UD {
                    return;
                }
                let ptr = cqe.user_data() as *mut Box<[f32]>;
                let result = unsafe { *Box::from_raw(ptr) };
                println!("Received: len={}, first={}", result.len(), result[0]);
                results.push(result);
            }

            let clones = (
                gpu.clone(),
                results[0].clone(),
                results[1].clone(),
                ring_fd,
            );

            // asynchronous invoking of gpu computation
            thread::spawn(move || {
                gpu::run_gpu(clones.0, clones.1, clones.2, clones.3).unwrap();
            });
        }
    });

    for _ in 1..=5 {
        thread::spawn(move || {
            for _ in 0..10 {
                let mut vector: Box<[f32]> = vec![0.00; ARRAY_SIZE].into_boxed_slice();
                
                // imitation of data receiving from somewhere
                for i in 0..ARRAY_SIZE {
                    vector[i] = rand::random_range(-10000.00..10001.00);
                }

                let ptr = Box::into_raw(Box::new(vector)) as u64;
                let mut msg_ring = IoUring::new(1).unwrap();
                let msg_e = opcode::MsgRingData::new(
                    types::Fd(listener_fd),
                    0,   
                    ptr,  
                    None,
                )
                .build();

                unsafe {
                    msg_ring.submission().push(&msg_e).ok();
                }
                msg_ring.submit().unwrap();
            }
        });
    }

    Ok(())
}
