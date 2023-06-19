use toml::Table;
use crate::xmitt::nuttall_pulse;
use crate::F;

pub fn load_xmission(file_name: &str) -> Table {
    let transmission = file_name.parse::<Table>().unwrap();
    let fc = transmission["xmitt"]["fc"].as_float().unwrap() as F;
    let pulse = match transmission["xmitt"]["name"].as_str().unwrap() {
        "nuttall" => Some(nuttall_pulse(fc)),
        _ => None
    };
    transmission
}


