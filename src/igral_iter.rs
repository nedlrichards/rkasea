use std::cmp;
use ndarray::{Array1, Axis, concatenate, s};
use std::thread;
use crossbeam_channel::{Receiver, Sender, bounded};
use crate::F;

//Chunk standardized the size of arrays that come out of an iterator
struct Chunk {
    chunk_size: usize,
    source: Counter,
    remanant: Array1<F>,
}

impl Chunk {
    fn new(chunk_size: usize, source: Counter) -> Chunk {
        Chunk{ chunk_size: chunk_size, source: source, remanant: Array1::ones(0) }
    }
}

impl Iterator for Chunk {
    type Item = Array1<F>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.remanant.len() < self.chunk_size {
            match self.source.next() {
                None => break,
                Some(vals) => {
                self.remanant = concatenate(Axis(0), &[vals.view(), self.remanant.view()]).to_owned().unwrap();
                },
            }
        }

        if self.remanant.len() == 0 {
            None
        }
        else if self.remanant.len() <= self.chunk_size {
            let result = self.remanant.clone();
            self.remanant = Array1::ones(0);
            Some(result)
        }
        else {
            let result = self.remanant.slice(s![..self.chunk_size]).to_owned();
            self.remanant = self.remanant.slice(s![self.chunk_size..]).to_owned();
            Some(result)
        }
    }

}


    fn reduce(ind: F, accum: &mut Array1<F>) {
        let a_ind = (ind / 10) as usize;
        let a_val = ind % 10;
        accum[a_ind] += a_val;
    }


    fn operation(input: Array1<F>) -> Array1<F> {
        let mut accum: Array1<F> = Array1::zeros(10);
        input.map(|&val| reduce(val, &mut accum));
        accum
    }

    let (s, r) = bounded(5);

    let (sout, rout) = bounded(5);

    thread::spawn(move || {
        for c in chunk {
            s.send(c).unwrap();
        }
        drop(s);
    });

    fn thread_op(rcvr: Receiver<Array1<F>>, sndr: Sender<Array1<F>>) {
        thread::spawn(move || {
            let mut accum: Array1<F> = Array1::zeros(10);
            for received in rcvr.iter() {
                accum = &accum + operation(received);
                thread::sleep(Duration::new(0, 50000000));
            }
            sndr.send(accum).unwrap();
        });
    }

    let mut accum: Array1<F> = Array1::zeros(10);
    for r in rout.iter() {
        accum = &accum + r;
    }

}
