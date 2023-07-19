use ndarray::{Array1, arr1};
use hdf5::{File, Result};
use toml::Table;
use std::path::Path;
use std::fs;

use crate::F;
use crate::xmission::{nuttall_pulse, Pulse};
use crate::surface::geometry::Surface1d;
use crate::greens::dist_img;
use crate::surface::bound_axes::bound_axes_x;


pub struct IO {
    file_name: String,
    hdf_file: File,
    toml_table: Table,
}

#[derive(Clone)]
pub struct Static {
    pub r_src: Array1<F>,
    pub r_rcr: Array1<F>,
    pub c: F,
    pub tau_0: F,
    pub duration: F,
    pub pulse: Pulse,
}


impl IO {
    pub fn write_setup(&self, xmission: &Pulse) -> Result<()> {

        let group = self.hdf_file.create_group("xmission")?;
        let builder = group.new_dataset_builder();
        let ds = builder.with_data(&xmission.signal).create("xmission")?;
        let attr_fs = &ds.new_attr::<F>().create("fs")?;
        let attr_fc = &ds.new_attr::<F>().create("fc")?;
        //let attr_name = &ds.new_attr::<String>().create("type").unwrap();
        attr_fs.write_scalar(&xmission.fs)?;
        attr_fc.write_scalar(&xmission.fc)?;

        Ok(())
        //group.new_dataset::<F>().shape(xmission.signal.len()).create("signal").unwrap().write(&xmission.signal);
    }

   pub fn load_setup(&self) -> Static {
        let zsrc = self.toml_table["static"]["zsrc"].as_float().unwrap() as F;
        let r_src: Array1<F> = arr1(&[0.0, 0.0, zsrc]);


        let zrcr = self.toml_table["static"]["zrcr"].as_float().unwrap() as F;
        let dr = self.toml_table["static"]["dr"].as_float().unwrap() as F;
        let theta = self.toml_table["static"]["theta"].as_float().unwrap() as F;
        let r_rcr: Array1<F> = arr1(&[dr * theta.cos(), dr * theta.sin(), zrcr]);

        let c = self.toml_table["static"]["c"].as_float().unwrap() as F;
        let t_pad = self.toml_table["static"]["t_pad"].as_float().unwrap() as F;
        let d_img = dist_img(&r_src, &r_rcr);
        let tau_0 = d_img / c - t_pad;
        let duration = self.toml_table["static"]["duration"].as_float().unwrap() as F;


        let fc = self.toml_table["static"]["fc"].as_float().unwrap() as F;
        let pulse = nuttall_pulse(fc);

        Static{
            r_src: r_src,
            r_rcr: r_rcr,
            c: c,
            tau_0: tau_0,
            duration: duration,
            pulse: pulse}
        }

   pub fn load_surface(&self, stat: &Static) -> Surface1d {
        let surface_type = self.toml_table["surface"]["type"].to_string();
        let time_step = self.toml_table["surface"]["time_step"].as_float().unwrap() as F;
        let num_steps = self.toml_table["surface"]["num_steps"].as_integer().unwrap() as i64;

        // TODO: Remove hardcode
        let offset: F = 0.5;
        let decimation = self.toml_table["static"]["decimation"].as_float().unwrap() as F;
        let dx = stat.c / (decimation * stat.pulse.fc);

        let (x0, x_n) = bound_axes_x(&stat.r_src, &stat.r_rcr, stat.tau_0 * stat.c, offset);
        let num_x = ((x_n - x0) / dx) as i64;

        Surface1d {
            name:surface_type,
            dx: dx,
            x0: x0,
            num_x: num_x,
        }
   }
}


pub fn build_io(toml_fn: &str) -> IO {
    let toml_table = fs::read_to_string(&toml_fn).unwrap().parse::<Table>().unwrap();
    //hdf file name is same as toml, with new extension
    let toml_path = Path::new(toml_fn);
    let hdf_parent = toml_path.parent().unwrap();
    let mut hdf_fn = toml_path.file_stem().unwrap().to_os_string();
    hdf_fn.push(".hd");
    let hdf_fn =  hdf_parent.join(hdf_fn);
    let hdf_file = File::create(&hdf_fn).unwrap();
    IO {
        file_name: String::from(toml_fn),
        hdf_file: hdf_file,
        toml_table: toml_table,
    }
}
