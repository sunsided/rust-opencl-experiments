#![allow(dead_code)]

use ocl::core::DeviceInfoResult;
use ocl::{Context, Device, Program};

// Requires the cl_khr_priority_queue extension
const EXTENSION_PRIORITY_QUEUE_SOURCE: &str = include_str!("topk_priority_queue_ext.cl");

const MANUAL_PRIORITY_QUEUE_SOURCE: &str = include_str!("topk_manual.cl");

pub fn build_priority_queue_program(device: &Device, context: &Context) -> ocl::Result<Program> {
    // Check if the device supports the cl_khr_priority_queue extension
    match device.info(ocl::enums::DeviceInfo::Extensions).unwrap() {
        DeviceInfoResult::Extensions(extensions) => {
            if !extensions.contains("cl_khr_priority_queue") {
                eprintln!("👎 Device does not support the cl_khr_priority_queue extension.");

                Program::builder()
                    .devices(device)
                    .src(MANUAL_PRIORITY_QUEUE_SOURCE)
                    .build(context)
            } else {
                println!("🎉 Device support the cl_khr_priority_queue extension!");
                let _program = Program::builder()
                    .devices(device)
                    .src(EXTENSION_PRIORITY_QUEUE_SOURCE)
                    .build(context);

                todo!("Ensure Kernel has the correct inputs");
            }
        }
        _ => unreachable!(),
    }
}
