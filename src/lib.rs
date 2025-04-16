extern crate blas_src;

use std::borrow::Borrow;
use std::ops::Div;

use ndarray::Array2;
use ndarray::Ix2;
use num_traits::real::Real;
use numpy::ndarray::Zip;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::types::{PyAnyMethods, PyDict};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use rand::rng;
use rand::Rng;

trait ClampMin: Real + Borrow<Self> {
    fn clamp_min(self, min: Self) -> Self;
    fn safe_divide(self, divisor: &Self, value: &Self, eps: Option<Self>) -> Self
    where
        Self: Sized;
}
macro_rules! impl_clamp_min {
    ($($t:ty)*) => ($(
        impl ClampMin for $t {
            fn clamp_min(self, min: Self) -> Self {
                if self < min {
                    min
                } else {
                    self
                }
            }
            fn safe_divide(self, divider: &Self,value : &Self,  eps:Option<Self>) -> Self {
                if divider.abs() < eps.unwrap_or(Self::from(1e-8)) {
                    return *value;
                } else {
                    self/divider
                }
            }
        }
    )*)
}
impl_clamp_min! { f32 f64 }

use ndarray::Data;
use ndarray::Dimension;
use num_traits::{FromPrimitive, Zero};

fn norm_l2<I, A>(x: I) -> A
where
    I: Iterator<Item = A>,
    A: ClampMin + Zero + Borrow<A>,
{
    x.map(|x| x.powi(2))
        .reduce(|acc, x| acc + x)
        .unwrap()
        .clamp_min(A::zero())
        .sqrt()
}

fn znnc_backend_single_data_generic<A, D>(
    row: &ndarray::ArrayView1<'_, A>,
    y: &ndarray::ArrayView1<'_, A>,
    eps: A,
) -> A
where
    A: ClampMin + FromPrimitive + Div<Output = A> + Borrow<A> + 'static,
    D: Dimension,
{
    let means = (row.mean().unwrap(), y.mean().unwrap());
    let row_c = row.map(|x| (*x - means.0));
    let y_c = y.map(|x| (*x - means.1));
    let row_c_norm = norm_l2(row_c.iter().map(|&x| x));
    let y_c_norm = norm_l2(y_c.iter().map(|&x| x));
    if row_c_norm < eps && y_c_norm < eps {
        return if row == y { A::one() } else { A::zero() };
    } else if row_c_norm < eps || y_c_norm < eps {
        return A::zero();
    }

    row_c
        .map(|&x| x / row_c_norm)
        .dot(&y_c.map(|&x| x / y_c_norm))
}

fn zncc_f64_backend(
    x: &ndarray::ArrayView2<f64>,
    y: &ndarray::ArrayView1<f64>,
) -> ndarray::Array1<f64> {
    Zip::from(x.rows())
        .par_map_collect(|row| znnc_backend_single_data_generic::<f64, Ix2>(&row, y, 1e-8))
}

fn bzncc_f64_backend(
    x: &ndarray::ArrayView2<f64>,
    y: &ndarray::ArrayView2<f64>,
) -> ndarray::Array2<f64> {
    let mut result = Array2::<f64>::zeros((x.nrows(), y.nrows()));
    Zip::indexed(&mut result).par_map_collect(|index, el| {
        *el = znnc_backend_single_data_generic::<f64, Ix2>(&x.row(index.0), &y.row(index.1), 1e-8);
    });
    result
}

fn aminmax(data: &ndarray::ArrayView2<f64>) -> (ndarray::Array1<f64>, ndarray::Array1<f64>) {
    let mut min_vals = data.row(0).to_owned();
    let mut max_vals = data.row(0).to_owned();
    for row in data.rows() {
        for (i, &val) in row.iter().enumerate() {
            if val < min_vals[i] {
                min_vals[i] = val;
            }
            if val > max_vals[i] {
                max_vals[i] = val;
            }
        }
    }
    return (min_vals, max_vals);
}

struct Clustering {
    k_clusters: usize,
    centroids: ndarray::Array2<f64>,
    clusters: ndarray::Array1<usize>,
    loop_count: usize,
}

impl Clustering {
    fn predict_func(
        data: &ndarray::ArrayView2<f64>,
        centroids: &ndarray::ArrayView2<f64>,
    ) -> (ndarray::Array2<f64>, ndarray::Array1<usize>) {
        let similarities = bzncc_f64_backend(&data.view(), &centroids.view()); // (x.rows(), k_clusters)
        let clusters = Zip::from(similarities.rows()).par_map_collect(|x| {
            x.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        });
        return (similarities, clusters);
    }

    fn predict(
        &self,
        data: &ndarray::ArrayView2<f64>,
    ) -> (ndarray::Array2<f64>, ndarray::Array1<usize>) {
        Self::predict_func(data, &self.centroids.view())
    }

    fn compute_inertia(
        data: &ndarray::ArrayView2<f64>,
        centroids: &ndarray::ArrayView2<f64>,
        clusters: &ndarray::ArrayView1<usize>,
    ) -> f64 {
        Zip::indexed(data.rows())
            .par_map_collect(|idx, row| {
                norm_l2(
                    centroids
                        .row(clusters[idx])
                        .iter()
                        .zip(row.iter())
                        .map(|(a, b)| (a - b)),
                )
            })
            .sum()
    }

    fn compute_centroids(
        data: &ndarray::ArrayView2<f64>,
        clusters: &ndarray::Array1<usize>,
        cluster_counts: &mut Vec<f64>,
    ) -> ndarray::Array2<f64> {
        cluster_counts.iter_mut().for_each(|x| *x = 0.0);
        for &cluster in clusters {
            cluster_counts[cluster] += 1.0;
        }
        Zip::indexed(data.rows()).par_fold(
            || ndarray::Array2::<f64>::zeros((cluster_counts.len(), data.ncols())),
            |mut acc_centroids, idx, row| {
                acc_centroids
                    .row_mut(clusters[idx])
                    .zip_mut_with(&row, |curr_val, x| {
                        if cluster_counts[clusters[idx]] != 0.0 {
                            *curr_val += x / cluster_counts[clusters[idx]]
                        }
                    });
                acc_centroids
            },
            |acc, x| acc + x,
        )
    }

    fn init_random_centroids(
        k_clusters: usize,
        dimension: usize,
        min_vals: &ndarray::Array1<f64>,
        max_vals: &ndarray::Array1<f64>,
    ) -> ndarray::Array2<f64> {
        let mut centroids = ndarray::Array2::<f64>::zeros((k_clusters, dimension));
        let mut rng = rand::rng();
        for mut centroid in centroids.rows_mut() {
            for (i, val) in centroid.iter_mut().enumerate() {
                *val = rng.random_range(min_vals[i]..=max_vals[i]);
            }
        }
        centroids
    }

    fn new(
        k_clusters: usize,
        data: ndarray::ArrayView2<f64>,
        n_max_iter: Option<usize>,
        inertia_eps: Option<f64>,
    ) -> Self {
        let mut inertia_roll = (None, None);
        let mut cluster_counts = vec![0.; k_clusters];

        let (min_vals, max_vals) = aminmax(&data.view());
        let mut centroids =
            Clustering::init_random_centroids(k_clusters, data.ncols(), &min_vals, &max_vals);
        let mut clusters;
        let mut loop_count: usize = 0;
        let mut rng = rand::rng();
        let mut weights;
        let mut weights_sum: f64;
        let mut new_centroid;
        let mut n_valid_clusters;
        loop {
            clusters = Clustering::predict_func(&data, &centroids.view()).1;
            centroids = Clustering::compute_centroids(&data, &clusters, &mut cluster_counts);

            for (i, &count) in cluster_counts.iter().enumerate() {
                if count == 0.0 {
                    n_valid_clusters = cluster_counts.iter().filter(|&&count| count > 0.0).count();
                    weights = (0..n_valid_clusters)
                        .map(|_| rng.random_range(0.0..1.0))
                        .collect::<Vec<f64>>();
                    weights_sum = weights.iter().sum();
                    new_centroid = centroids
                        .rows()
                        .into_iter()
                        .zip(cluster_counts.iter())
                        .filter(|(_, &count)| count > 0.0)
                        .map(|(row, _)| row)
                        .enumerate()
                        .fold(
                            ndarray::Array1::<f64>::zeros(data.ncols()),
                            |acc, (k, x)| acc + weights[k] * &x,
                        )
                        .map(|x| x / weights_sum);
                    centroids.row_mut(i).assign(&new_centroid);
                }
            }
            if let Some(eps) = inertia_eps {
                let inertia =
                    Clustering::compute_inertia(&data, &centroids.view(), &clusters.view());

                match inertia_roll {
                    (None, None) => {
                        inertia_roll.0 = Some(inertia);
                    }
                    (Some(d), None) => {
                        inertia_roll = (Some(inertia), Some(d));
                    }
                    (Some(d), Some(d2)) => {
                        if (d - d2).abs() / d2 < eps {
                            break;
                        }
                        inertia_roll = (Some(inertia), Some(d));
                    }
                    _ => {}
                }
            }
            if loop_count >= n_max_iter.unwrap_or(300) {
                break;
            }
            loop_count += 1;
        }

        Self {
            k_clusters: k_clusters,
            centroids: centroids,
            loop_count: loop_count,
            clusters: clusters,
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn kmax_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    fn zncc<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        zncc_f64_backend(&x.as_array(), &y.as_array()).into_pyarray(py)
    }
    #[pyfn(m)]
    fn bzncc<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        bzncc_f64_backend(&x.as_array(), &y.as_array()).into_pyarray(py)
    }

    #[pyfn(m)]
    fn kmax<'py>(
        py: Python<'py>,
        k_clusters: usize,
        x: PyReadonlyArray2<'py, f64>,
        n_max_iter: Option<usize>,
        inertia_eps: Option<f64>,
    ) -> Bound<'py, PyDict> {
        let data = x.as_array();
        let clustering = Clustering::new(k_clusters, data, n_max_iter, inertia_eps);
        let dict = PyDict::new(py);
        dict.set_item("k_clusters", clustering.k_clusters).unwrap();
        dict.set_item("centroids", clustering.centroids.into_pyarray(py))
            .unwrap();
        dict.set_item("clusters", clustering.clusters.into_pyarray(py))
            .unwrap();
        dict.set_item("loop_count", clustering.loop_count).unwrap();
        dict
    }

    Ok(())
}
