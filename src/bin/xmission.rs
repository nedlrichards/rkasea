use rkasea::load_xmission::load_xmission;
use std::{fs, env};

fn main() {

    let args: Vec<String> = env::args().collect();
    let file_name = &args[1];

    let contents = fs::read_to_string(file_name)
        .expect("Should have been able to read the file");

    let test = load_xmission(&contents);
    println!("{}", test["t_f"]);
}
