#ifndef BASEDNN_H
#define BASEDNN_H

#include <stdio.h>

#include "tensor.h"
#include "ops.h"
#include "registry.h"
#include "layer.h"
#include "network.h"
#include "optimizer.h"

#ifdef HAS_WGPU
#include "../../backend/wgpu/wgpu_backend.h"
#endif

// Initialize the registry with built-in layers, losses, and optimizers
// Call this once at the start of your program
static inline void basednn_init() {
    registry_init();
    
#ifdef HAS_WGPU
    if (wgpu_init() == 0 && wgpu_available()) {
        wgpu_register_ops();
        printf("GPU acceleration enabled (wgpu-native)\n");
    } else {
        printf("Using CPU backend\n");
    }
#else
    printf("Using CPU backend\n");
#endif
}

// Cleanup registry resources
// Call this at the end of your program
static inline void basednn_cleanup() {
#ifdef HAS_WGPU
    wgpu_cleanup();
#endif
    registry_cleanup();
}

#endif