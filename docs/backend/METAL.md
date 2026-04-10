# Metal Backend

The Metal backend enables GPU-accelerated inference on macOS using Apple's Metal framework.

## eGPU / External GPU Support (USB4, Thunderbolt)

On macOS, llama.cpp uses the Metal backend for GPU-accelerated inference. By default it selects the first enumerated Metal device (usually the built-in GPU). To use an **external GPU (eGPU)** connected via USB4 or Thunderbolt, use the `GGML_METAL_DEVICE_INDEX` environment variable.

### Discovering available devices

At startup, llama.cpp prints all available Metal devices and their indices:

```
ggml_metal_device_init: available Metal devices:
ggml_metal_device_init:   [0] Apple M2 Pro (built-in (internal))
ggml_metal_device_init:   [1] AMD Radeon RX 6800 XT (external (eGPU))
```

### Selecting a specific GPU

```bash
# Use the eGPU at index 1:
GGML_METAL_DEVICE_INDEX=1 ./llama-cli -m model.gguf -p "Hello"
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GGML_METAL_DEVICE_INDEX` | `0` | Physical index of the Metal device to use (as listed at startup). Set to `1` for a second GPU / eGPU. |
| `GGML_METAL_DEVICES` | `1` | Number of logical Metal devices to expose. Combine with `GGML_METAL_DEVICE_INDEX` for multi-GPU setups spanning both internal and external GPUs. |

### Multi-GPU (internal + eGPU)

```bash
# Expose 2 logical devices: logical 0 -> physical 0 (internal), logical 1 -> physical 1 (eGPU)
GGML_METAL_DEVICES=2 GGML_METAL_DEVICE_INDEX=0 ./llama-cli -m model.gguf --n-gpu-layers 99
```

### Notes

- The eGPU must be connected and recognized by macOS before launching llama.cpp.
- eGPUs are shown with location `external (eGPU)` in the device list.
- Performance over Thunderbolt/USB4 may be lower than a native PCIe GPU due to bandwidth limitations; this is a hardware constraint, not a software one.
- This feature is macOS-only. On iOS/tvOS/visionOS only one Metal device is available and the device index is ignored.