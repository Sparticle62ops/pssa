//! Runtime device selection for training. Auto preserves the existing fallback.
use crate::backend::{Device, WgpuContext};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrainingBackend {
    #[default]
    Auto,
    Cpu,
    WebGpu,
    Cuda,
}

impl TrainingBackend {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "webgpu" | "wgpu" => Ok(Self::WebGpu),
            "cuda" if cfg!(feature = "cuda") => Ok(Self::Cuda),
            "cuda" => Err("CUDA requires a binary built with --features cuda".into()),
            _ => Err("--backend must be auto, cpu, webgpu (or wgpu), or cuda".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::WebGpu => "webgpu",
            Self::Cuda => "cuda",
        }
    }

    pub(crate) fn device(self) -> Result<Device, String> {
        match self {
            Self::Auto => Device::try_gpu(),
            Self::Cpu => Ok(Device::Cpu),
            Self::WebGpu => WgpuContext::init_blocking().map(Device::Gpu),
            #[cfg(feature = "cuda")]
            Self::Cuda => crate::cuda::CudaContext::init().map(Device::Cuda),
            #[cfg(not(feature = "cuda"))]
            Self::Cuda => Err("CUDA requires a binary built with --features cuda".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_selection_is_explicit_and_cpu_does_not_probe_a_gpu() {
        assert_eq!(TrainingBackend::default(), TrainingBackend::Auto);
        assert!(matches!(TrainingBackend::Cpu.device(), Ok(Device::Cpu)));
        for value in ["auto", "cpu", "webgpu"] {
            assert_eq!(TrainingBackend::parse(value).unwrap().as_str(), value);
        }
        assert_eq!(TrainingBackend::parse("wgpu"), Ok(TrainingBackend::WebGpu));
        assert!(TrainingBackend::parse("tpu").is_err());
        assert_eq!(
            TrainingBackend::parse("cuda").is_ok(),
            cfg!(feature = "cuda")
        );
    }
}
