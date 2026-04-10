#import "ggml-metal-device.h"

#import "ggml-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <stdatomic.h>

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetalX (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

//
// MTLFunctionConstantValues wrapper
//
struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;
};
ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
}

//
// MTLComputePipelineState wrapper
//
struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;
};
ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj  =*/ nil,
    };

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.maxTotalThreadsPerThreadgroup;
}

struct ggml_metal_library {
    id<MTLLibrary> obj;
    id<MTLDevice> device;

    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines

    NSLock * lock;
};
ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLLibrary> library = nil;
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    //
    // TODO: move to a function
    {
        const int64_t t_start = ggml_time_us();

        NSError * error = nil;
        NSString * src = nil;

#if GGML_METAL_EMBED_LIBRARY
        GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

        extern const char ggml_metallib_start[];
        extern const char ggml_metallib_end[];

        src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];
#else

#ifdef SWIFT_PACKAGE
        NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

        NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
        if (path_lib == nil) {
            // Try to find the resource in the directory where the current binary located.
            NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
            NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

            NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
            if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

                NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
                if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                    // Optionally, if this is a symlink, try to resolve it.
                    path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                    if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                        // It is a relative path, adding the binary directory as directory prefix.
                        path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                    }
                    if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                        // Link to the resource could not be resolved.
                        path_lib_default = nil;
                    } else {
                        GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                    }
                }
            } else {
                // The resource couldn't be found in the binary's directory.
                path_lib_default = nil;
            }

            path_lib = path_lib_default;
        }

        if (path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            library = [device newLibraryWithURL:libURL error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        } else {
            GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];

            GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        }
#endif

        if (!library) {
            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                if (ggml_metal_device_get_props(dev)->has_bfloat) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
                }

                if (ggml_metal_device_get_props(dev)->has_tensor) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"];
                }

#if GGML_METAL_EMBED_LIBRARY
                [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;

                //[options setFastMathEnabled:false];

                library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return nil;
                }

#if !__has_feature(objc_arc)
                [options release];
#endif
            }
        }

#if GGML_METAL_EMBED_LIBRARY
        [src release];
#endif // GGML_METAL_EMBED_LIBRARY

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));

    res->obj       = library;
    res->device    = device;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    return res;
}
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose) {
    if (source == NULL) {
        GGML_LOG_ERROR("%s: source is NULL\n", __func__);
        return NULL;
    }

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLLibrary> library = nil;
    NSError * error = nil;

    const int64_t t_start = ggml_time_us();

    NSString * src = [[NSString alloc] initWithBytes:source
                                              length:strlen(source)
                                            encoding:NSUTF8StringEncoding];
    if (!src) {
        GGML_LOG_ERROR("%s: failed to create NSString from source\n", __func__);
        return NULL;
    }

    @autoreleasepool {
        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = prep;

        library = [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            if (verbose) {
                GGML_LOG_ERROR("%s: error compiling source: %s\n", __func__, [[error description] UTF8String]);
            } else {
                GGML_LOG_ERROR("%s: error compiling source\n", __func__);
            }
            library = nil;
        }

        [options release];
    }

    [src release];

    if (!library) {
        if (verbose) {
            GGML_LOG_ERROR("%s: failed to create Metal library from source\n", __func__);
        }

        return NULL;
    }

    if (verbose) {
        GGML_LOG_INFO("%s: compiled in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    if (!res) {
        GGML_LOG_ERROR("%s: calloc failed\n", __func__);
        return NULL;
    }

    res->obj       = library;
    res->device    = device;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    return res;
}

void ggml_metal_library_free(ggml_metal_library_t lib) {
    if (!lib) {
        return;
    }

    if (lib->obj) {
        [lib->obj release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    [lib->lock release];

    free(lib);
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    [lib->lock lock];

    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);

    [lib->lock unlock];

    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);
    if (res.pipeline) {
        [lib->lock unlock];

        return res;
    }

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        id<MTLFunction> mtl_function;
        if (!cv) {
            mtl_function = [lib->obj newFunctionWithName:base_func];
        } else {
            mtl_function = [lib->obj newFunctionWithName:base_func constantValues:cv->obj error:&error];
        }
        if (!mtl_function) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        id<MTLComputePipelineState> obj = [lib->device newComputePipelineStateWithFunction:mtl_function error:&error];

        [mtl_function release];

        if (!obj) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to create pipeline state: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name,
                (void *) obj,
                (int)    obj.maxTotalThreadsPerThreadgroup,
                (int)    obj.threadExecutionWidth);

        if (obj.maxTotalThreadsPerThreadgroup == 0 || obj.threadExecutionWidth == 0) {
            [obj release];

            [lib->lock unlock];

            GGML_LOG_ERROR("%s: incompatible pipeline %s\n", __func__, name);

            return res;
        }

        res.pipeline = ggml_metal_pipeline_init();
        res.pipeline->obj = obj;

        ggml_metal_pipelines_add(lib->pipelines, name, res.pipeline);
    }

    [lib->lock unlock];

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//
struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;
};
ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    if (concurrent) {
        res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        res->obj = [cmd_buf computeCommandEncoder];
    }

    [res->obj retain];

    return res;
}

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    [encoder->obj release];
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    [encoder->obj setComputePipelineState:pipeline.pipeline->obj];
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    [encoder->obj setBytes:data length:size atIndex:idx];
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    [encoder->obj endEncoding];
}

struct ggml_metal_device {
    id<MTLDevice> mtl_device;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_rsets_t rsets;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;
};

//
// MTLResidenceSet wrapper
//
struct ggml_metal_rsets {
    NSLock * lock;

    NSMutableArray * data;

    // number of seconds since the last graph computation
    // keep the residency sets wired for that amount of time to avoid being collected by the OS
    int keep_alive_s;

    // background heartbeat thread to keep the residency sets alive
    atomic_bool d_stop;
    atomic_int  d_loop;

    dispatch_group_t d_group;
};
ggml_metal_rsets_t ggml_metal_rsets_init(void) {
    ggml_metal_rsets_t res = calloc(1, sizeof(struct ggml_metal_rsets));

    res->lock = [[NSLock alloc] init];
    res->data = [[NSMutableArray alloc] init];

    // by default keep the memory wired for 3 minutes
    res->keep_alive_s = 3*60;

    const char * GGML_METAL_RESIDENCY_KEEP_ALIVE_S = getenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S");
    if (GGML_METAL_RESIDENCY_KEEP_ALIVE_S) {
        res->keep_alive_s = atoi(GGML_METAL_RESIDENCY_KEEP_ALIVE_S);
    }

    if (res->keep_alive_s <= 0) {
        res->keep_alive_s = 3*60;
    }

    GGML_LOG_INFO("%s: creating a residency set collection (keep_alive = %d s)\n", __func__, res->keep_alive_s);

    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);
    atomic_store_explicit(&res->d_loop, 2*res->keep_alive_s, memory_order_relaxed);

    res->d_group = dispatch_group_create();

    // start a background thread that periodically requests residency for all the currently active sets in the collection
    // the requests stop after a certain amount of time (keep_alive_s) of inactivity
    dispatch_queue_t d_queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_group_async(res->d_group, d_queue, ^{
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {
                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {
                      [res->lock lock];

                      for (int i = 0; i < (int) res->data.count; ++i) {
                          [res->data[i] requestResidency];
                      }

                      atomic_fetch_sub_explicit(&res->d_loop, 1, memory_order_relaxed);

                      [res->lock unlock];
                  }

                  // half a second
                  usleep(500 * 1000);
              }
        }
#endif
    });

    return res;
}

void ggml_metal_rsets_free(ggml_metal_rsets_t rsets) {
    if (rsets == NULL) {
        return;
    }

    // note: if you hit this assert, most likely you haven't deallocated all Metal resources before exiting
    GGML_ASSERT([rsets->data count] == 0);

    atomic_store_explicit(&rsets->d_stop, true, memory_order_relaxed);

    dispatch_group_wait(rsets->d_group, DISPATCH_TIME_FOREVER);
    dispatch_release(rsets->d_group);

    [rsets->data release];
    [rsets->lock release];

    free(rsets);
}

ggml_metal_device_t ggml_metal_device_init(int device) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    if (dev->mtl_device == nil) {
#if TARGET_OS_OSX
        // Enumerate all Metal devices so the user can select an eGPU connected
        // via USB4/Thunderbolt. Inspired by tinygrad's device selection approach.
        // Set GGML_METAL_DEVICE_INDEX=N to use the N-th device in the list
        // (devices are printed at startup). Default is 0 (first enumerated device).
        {
            const char * env_device_index = getenv("GGML_METAL_DEVICE_INDEX");
            const int device_index_offset = env_device_index ? atoi(env_device_index) : 0;
            const int physical_index = device_index_offset + device;

            NSArray<id<MTLDevice>> * all_devices = MTLCopyAllDevices();

            // Log all available Metal devices so users can discover their eGPU index
            GGML_LOG_INFO("%s: available Metal devices:\n", __func__);
            for (int i = 0; i < (int)[all_devices count]; ++i) {
                id<MTLDevice> d = all_devices[i];
                const char * location_str = "unspecified";
                if (@available(macOS 10.13, *)) {
                    switch (d.location) {
                        case MTLDeviceLocationBuiltIn:   location_str = "built-in (internal)"; break;
                        case MTLDeviceLocationSlot:      location_str = "slot";                break;
                        case MTLDeviceLocationExternal:  location_str = "external (eGPU)";    break;
                        default:                         location_str = "unspecified";          break;
                    }
                }
                GGML_LOG_INFO("%s:   [%d] %s (%s)\n", __func__, i, [[d name] UTF8String], location_str);
            }

            if (physical_index >= 0 && physical_index < (int)[all_devices count]) {
                // retain because MTLCopyAllDevices returns autoreleased objects
                dev->mtl_device = [all_devices[physical_index] retain];
                GGML_LOG_INFO("%s: using device [%d] %s (GGML_METAL_DEVICE_INDEX=%d)\n",
                        __func__, physical_index, [[dev->mtl_device name] UTF8String], device_index_offset);
            } else {
                GGML_LOG_WARN("%s: GGML_METAL_DEVICE_INDEX=%d out of range (%d device(s) available), "
                              "falling back to system default\n",
                        __func__, device_index_offset, (int)[all_devices count]);
                dev->mtl_device = MTLCreateSystemDefaultDevice();
            }

            [all_devices release];
        }
#else
        // iOS / tvOS / visionOS: only one Metal device available
        dev->mtl_device = MTLCreateSystemDefaultDevice();
#endif

        if (dev->mtl_device) {
            dev->mtl_queue = [dev->mtl_device newCommandQueue];
            if (dev->mtl_queue == nil) {
                GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            }

            dev->addr_virt = 0x000000400ULL;

            dev->props.device = device;
            dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

            dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;

            dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
            dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
            if (getenv("GGML_METAL_BF16_DISABLE") != NULL) {
                dev->props.has_bfloat = false;
            }

            dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
            if (getenv("GGML_METAL_TENSOR_DISABLE") != NULL) {
                dev->props.has_tensor = false;
            }

            // note: disable the tensor API by default for old chips because with the current implementation it is not useful
            // - M2 Ultra:   ~5% slower
            // - M4, M4 Max: no significant difference
            //
            // TODO: try to update the tensor API kernels to at least match the simdgroup performance
            if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
                ![[dev->mtl_device name] containsString:@"M5"] &&
                ![[dev->mtl_device name] containsString:@"M6"] &&
                ![[dev->mtl_device name] containsString:@"A19"] &&
                ![[dev->mtl_device name] containsString:@"A20"]) {
                GGML_LOG_WARN("%s: tensor API disabled for pre-M5 and pre-A19 devices\n", __func__);
                dev->props.has_tensor = false;
            }

            // double-check that the tensor API compiles
            if (dev->props.has_tensor) {
                const char * src_tensor_f16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for f16 support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_f16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                    dev->props.has_tensor = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                        dev->props.has_tensor = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            // try to compile a dummy kernel to determine if the tensor API is supported for bfloat
            if (dev->props.has_tensor && dev->props.has_bfloat) {
                const char * src_tensor_bf16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for bfloat support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_bf16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                    dev->props.has_bfloat = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                        dev->props.has_bfloat = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
            dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

            dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX
            if (@available(macOS 10.13, *)) {
                // For eGPU (external/removable devices), shared memory is preferable
                dev->props.use_shared_buffers |= (dev->mtl_device.location == MTLDeviceLocationExternal);
            }
#endif
            if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                dev->props.use_shared_buffers = false;
            }
            if (getenv("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                dev->props.use_shared_buffers = true;
            }

            dev->props.supports_gpu_family_apple7 = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

            dev->props.op_offload_min_batch_size  = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;
            dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;
            if (@available(macOS 10.12, iOS 16.0, *)) {
                dev->props.max_working_set_size   = dev->mtl_device.recommendedMaxWorkingSetSize;
            } else {
                dev->props.max_working_set_size   = dev->mtl_device.maxBufferLength;
            }

            snprintf(dev->props.name, sizeof(dev->props.name), "%s%d", "MTL", device);
            snprintf(dev->props.desc, sizeof(dev->props.desc), "%s", [[dev->mtl_device name] UTF8String]);

            dev->library = ggml_metal_library_init(dev);
            if (!dev->library) {
                GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
            }

            if (dev->props.use_residency_sets) {
                dev->rsets = ggml_metal_rsets_init();
            } else {
                dev->rsets = nil;
            }

            // print MTL GPU family:
            GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, dev->props.name);

            // determine max supported GPU family
            // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            {
                for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                        break;
                    }
                }
            }

            GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
            GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
            GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
            GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
            GGML_LOG_INFO("%s: has tensor            = %s\n", __func__, dev->props.has_tensor              ? "true" : "false");
            GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
            GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
            if (@available(macOS 10.12, iOS 16.0, *)) {
                GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
            }
#endif
        }
    }

    return dev;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    ggml_metal_rsets_free(dev->rsets);

    ggml_metal_library_free(dev->library);
    dev->library = NULL;

    if (dev->mtl_queue) {
        [dev->mtl_queue release];
        dev->mtl_queue = nil;
    }

    if (dev->mtl_device) {
        [dev->mtl_device release];
        dev->mtl_device = nil;
    }

    free(dev);
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data addObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_rm(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data removeObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    if (dev->rsets == NULL) {
        return;
    }

    atomic_store_explicit(&dev->rsets->d_loop, 2*dev->rsets->keep_alive_s, memory_order_relaxed);
}

struct ggml_metal_event {
    void * obj; // id<MTLEvent>

    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLEvent> event = (id<MTLEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add_explicit(&ev->value, 1, memory_order_relaxed) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLEvent> event = (id<MTLEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
}

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
    id<MTLEvent> event = [dev->mtl_device newEvent];

    ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));

    ev->obj = (__bridge void *)event;
    ev->value = 0;

    return ev;
}

void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLEvent> event = ev->obj;
    [event release];

    free(ev);

    GGML_UNUSED(dev);
}

void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLEvent> event = ev->obj;

        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];
        [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total = dev->mtl_device.recommendedMaxWorkingSetSize;
        *free  = *total - dev->mtl_device.currentAllocatedSize;
    } else {
        *free = 0;
        *total = 0;
    }
}