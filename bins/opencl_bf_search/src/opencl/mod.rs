mod dot_product;
mod dot_topk;
mod priority_queue;

use clap::ArgMatches;
use colored::Colorize;
pub use dot_product::build_dot_product_program;
use ocl::{Device, Platform};

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

pub struct OpenClDeviceSelection {
    pub platform: Platform,
    pub device: Device,
}

pub fn get_opencl_selection(matches: &ArgMatches) -> Option<OpenClDeviceSelection> {
    let platforms = Platform::list();
    if platforms.is_empty() {
        return None;
    }

    let pid = matches
        .get_one::<usize>("ocl-platform-id")
        .unwrap_or(&0)
        .to_owned();

    let did = matches
        .get_one::<usize>("ocl-device-id")
        .unwrap_or(&0)
        .to_owned();

    // This should already be validated by the argument parser.
    debug_assert!(pid < platforms.len(), "platform ID out of bounds");
    let platform = platforms[pid];

    let name = platform.name().unwrap_or(String::from("(unnamed)"));
    println!(
        "Using OpenCL platform {pid}: {name}",
        pid = pid.to_string().green(),
        name = name.green()
    );

    let devices = match Device::list_all(platform) {
        Ok(devices) => devices,
        Err(e) => {
            eprintln!("The selected platform has no available devices: {e}");
            return None;
        }
    };

    if did >= devices.len() {
        eprintln!(
            "Unable to select device {did} for platform {pid}",
            did = did.to_string().blue(),
            pid = pid.to_string().green()
        );
        return None;
    }

    let device = devices[did];
    let name = device.name().unwrap_or(String::from("(unnamed)"));
    println!(
        "Using OpenCL platform {pid}'s device {did}: {name}",
        pid = pid.to_string().green(),
        did = did.to_string().blue(),
        name = name.blue()
    );

    Some(OpenClDeviceSelection { platform, device })
}
