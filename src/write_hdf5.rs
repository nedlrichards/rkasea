#[cfg(feature = "blosc")]
use hdf5::filters::blosc_set_nthreads;
use hdf5::{File, Result};
use rkasea::xmitt::nuttall_pulse;
use rkasea::F;

fn write_hdf5(file_name: &str) -> Result<()> {
    let xmitt = nuttall_pulse(3e3, 20e3);
    let file = File::create(file_name)?; // open for writing
    let group = file.create_group("dir")?; // create a group
    //let dt: F = (t_a[t_a.len() - 1] - t_a[0]) / ((t_a.len() as F) - 1.0);

    group.new_dataset::<F>().shape(xmitt.dim()).create("pulse")?.write(&xmitt)?;

    //let builder = group.new_dataset_builder();
    //let ds = builder.with_data(&t_a).create("t_axis")?;
    //let attr_dt = &ds.new_attr::<F>().create("dt")?;
    //let attr_fs = &ds.new_attr::<F>().create("fs")?;
    //attr_dt.write_scalar(&dt)?;
    //attr_fs.write_scalar(&(1.0 / dt))?;

    Ok(())
}

fn main() -> Result<()> {
    let file_name = "xmission.h5";
    write_hdf5(&file_name)?;
    Ok(())
}
