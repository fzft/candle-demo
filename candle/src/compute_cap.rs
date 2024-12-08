use anyhow::Context;
use candle_core::cuda_backend::cudarc::driver;
use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
};
use candle_core::cuda_backend::cudarc::driver::CudaDevice;

pub fn get_compile_compute_cap() -> Result<usize, anyhow::Error> {
    std::env::var("CUDA_COMPUTE_CAP")
        .context("Could not retrieve compile time CUDA_COMPUTE_CAP")?
        .parse::<usize>()
        .context("Invalid CUDA_COMPUTE_CAP")
}

pub fn get_runtime_compute_cap() -> Result<usize, anyhow::Error> {
    driver::result::init().context("CUDA is not available")?;
    let device = CudaDevice::new(0).context("CUDA is not available")?;
    let major = device
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .context("Could not retrieve device compute capability major")?;
    let minor = device
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .context("Could not retrieve device compute capability minor")?;
    Ok((major * 10 + minor) as usize)
}

#[cfg(test)]
mod tests {
    use super::{get_compile_compute_cap, get_runtime_compute_cap};

    #[test]
    fn test_get_compile_compute_cap() {
        let cap = get_compile_compute_cap().unwrap();
        assert!(cap > 0);
    }

    #[test]
    fn test_get_runtime_compute_cap() {
        let cap = get_runtime_compute_cap().unwrap();
        assert!(cap > 0);
    }
}
