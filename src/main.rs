use std::error::Error;
mod gpu;
mod data_flow;
use io_uring::{IoUring, opcode, types};
use std::os::unix::io::AsRawFd;
use std::time::Duration;

const TIMEOUT_UD: u64 = 0x5449_4d45_4f55_54; // "TIMEOUT"

fn main() -> Result<(), Box<dyn Error>> {
    let gpu = gpu::init_gpu()?;

    let mut ring = IoUring::new(50).expect("io_uring failure");
    let ring_fd = ring.as_raw_fd();

    data_flow::data_flow(gpu, ring_fd).unwrap();


    // listening io_uring and receiving result from gpu
    'main: loop {
        let timeout = types::Timespec::from(Duration::from_secs(30));
        let timeout_e = opcode::Timeout::new(&timeout)
            .build()
            .user_data(TIMEOUT_UD);
        unsafe {
            ring.submission().push(&timeout_e).ok();
        }

        ring.submit_and_wait(1)?;
        let cq = ring.completion();

        for cqe in cq {
            if cqe.user_data() == TIMEOUT_UD {
                break 'main;
            }
            let ptr = cqe.user_data() as *mut f32;
            let value = unsafe { *Box::from_raw(ptr) };
            println!("GPU Result via io_uring: {}", value);
        }
    }

    Ok(())
}
