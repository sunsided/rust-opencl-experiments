use clap::{value_parser, Arg, ArgAction, ArgMatches, Command, ValueHint};
use std::path::PathBuf;

pub fn match_cli_arguments() -> ArgMatches {
    let command = Command::new("OpenCL Streaming Vector Dot Products")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Markus Mayer <widemeadows@gmail.com>")
        .about("Experiments using OpenCL for GPU-accelerated Streaming Vector Dot Products")
        .arg(
            Arg::new("ocl-list-platforms")
                .short('L')
                .long("list-platforms")
                .help("Lists available OpenCL devices")
                .help_heading("OpenCL")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("ocl-platform-id")
                .short('p')
                .long("platform")
                .value_name("PLATFORM_ID")
                .help("The ID of the platform to use")
                .help_heading("OpenCL")
                .num_args(1)
                .allow_negative_numbers(false)
                .value_parser(ocl_platform_valid),
        )
        .arg(
            Arg::new("ocl-device-id")
                .short('d')
                .long("device")
                .value_name("DEVICE_ID")
                .help("The ID of the selected platform's device to use")
                .help_heading("OpenCL")
                .num_args(1)
                .allow_negative_numbers(false)
                .value_parser(ocl_device_valid),
        )
        .arg(
            Arg::new("vector-db")
                .short('i')
                .long("input")
                .value_hint(ValueHint::FilePath)
                .value_name("FILE")
                .help("The vector database to load")
                .default_value("vectors.bin")
                .default_value_if("ocl-list-platforms", "true", None)
                .num_args(1)
                .value_parser(file_valid)
                .help_heading("Vector Database"),
        )
        .arg(
            Arg::new("max-vectors")
                .long("max-vecs")
                .value_name("COUNT")
                .help("The maximum number of vectors to load from the database")
                .default_value(if cfg!(debug_assertions) { "10000" } else { "0" })
                .hide_default_value(!cfg!(debug_assertions))
                .num_args(1)
                .allow_negative_numbers(false)
                .value_parser(num_vecs)
                .help_heading("Vector Database"),
        );

    command.get_matches()
}

fn filename_valid(s: &str) -> Result<PathBuf, String> {
    PathBuf::try_from(s).map_err(|_| String::from("The specified file name was invalid"))
}

fn file_valid(s: &str) -> Result<PathBuf, String> {
    let path_buf = filename_valid(s)?;
    let path = shellexpand_path(&path_buf)?;
    if path.is_file() {
        println!("The path is {path:?}");
        Ok(path)
    } else {
        Err(String::from("The specified file does not exist"))
    }
}

/// Expands environment variables and shell tokens such as `~` (for the home directory).
fn shellexpand_path(path: &PathBuf) -> Result<PathBuf, String> {
    let str = path.to_str().ok_or("Unable to represent path")?;
    let str = shellexpand::full(str).map_err(|_| "Unable to expand path components")?;
    Ok(PathBuf::from(str.to_string()))
}

fn ocl_platform_valid(s: &str) -> Result<usize, String> {
    let id: usize = s.parse().map_err(|e| format!("Invalid number: {e}"))?;
    let platforms = ocl::Platform::list();
    if platforms.is_empty() {
        return Err(String::from("No OpenCL platforms detected"));
    }

    if id >= platforms.len() {
        return Err(format!(
            "Only {len} platforms available, valid values in are in range 0 to {max}.",
            len = platforms.len(),
            max = platforms.len() - 1
        ));
    }

    Ok(id)
}

fn ocl_device_valid(s: &str) -> Result<usize, String> {
    let id: usize = s.parse().map_err(|e| format!("Invalid number: {e}"))?;

    let platforms = ocl::Platform::list();
    if platforms.is_empty() {
        return Err(String::from("No OpenCL platforms detected"));
    }

    if !platforms
        .iter()
        .flat_map(|p| ocl::Device::list_all(p))
        .any(|d| id < d.len())
    {
        return Err(String::from("No platform supports the specified ID"));
    }

    Ok(id)
}

fn num_vecs(s: &str) -> Result<usize, String> {
    const MIN_COUNT: usize = 256;
    let count: usize = s.parse().map_err(|e| format!("{e}"))?;
    if count < MIN_COUNT {
        Err(format!(
            "The number must be greater than or equal to {MIN_COUNT}"
        ))
    } else {
        Ok(count)
    }
}
