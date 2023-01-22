mod dot_product;
mod dot_topk;
mod priority_queue;

use colored::Colorize;
pub use dot_product::build_dot_product_program;
pub use dot_topk::build_dot_topk_program;
use ocl::{Device, Platform};
pub use priority_queue::build_priority_queue_program;

pub fn ocl_print_platforms() {
    let platforms = Platform::list();
    if platforms.is_empty() {
        eprintln!("No OpenCL platforms detected");
        return;
    }

    for (pid, platform) in platforms.iter().enumerate() {
        if pid > 0 {
            println!();
        }

        let name = platform.name().unwrap_or(String::from("(unnamed)")).green();
        let ocl_version = platform
            .version()
            .unwrap_or_else(|e| format!("(invalid platform version: {e})"));
        let profile = platform
            .profile()
            .unwrap_or_else(|e| format!("invalid platform profile: {e}"));
        println!(
            "{pid}: {name}, {ocl_version} ({profile})",
            pid = format!("{pid:04}").green()
        );

        match Device::list_all(platform) {
            Err(e) => println!("  Failed to enumerate devices: {e}"),
            Ok(devices) => {
                println!("      Number of devices: {count}", count = devices.len());
                for (did, device) in devices.iter().enumerate() {
                    let name = device
                        .name()
                        .unwrap_or_else(|e| format!("(invalid device name: {e})"))
                        .blue();
                    let ocl_version = platform
                        .version()
                        .unwrap_or_else(|e| format!("(invalid device version: {e})"));
                    println!("      {did}: {name}", did = format!("{did:04}").blue());
                    println!("            {ocl_version}");
                }
            }
        }
    }
}
