"""Check an actual CUDA operation, not just package import."""
import json
import torch
import torchvision


def main():
    info = {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if not info["cuda_available"]:
        print(json.dumps(info, indent=2))
        raise SystemExit("CUDA unavailable: check the WSL GPU driver and this Python environment.")
    x = torch.ones((32, 32), device="cuda")
    result = x @ x
    torch.cuda.synchronize()
    if not torch.equal(result, torch.full_like(result, 32)):
        raise RuntimeError("CUDA matrix multiplication returned an unexpected result")
    info.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                matmul_shape=list(result.shape), matmul_value=result[0, 0].item(), passed=True)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
