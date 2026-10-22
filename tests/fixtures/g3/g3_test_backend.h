// Original CPU-only behavioral substitute. Never install with native CUDA/cuFile headers.
#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaMemoryTypeDevice = 2, cudaHostRegisterDefault = 0;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
struct cudaUUID_t { char bytes[16]; };
struct cudaPointerAttributes { int type; };
inline int cudaSetDevice(int) { return 0; }
inline int cudaDeviceGetUuid(cudaUUID_t* uuid, int) { std::memset(uuid, 0, sizeof(*uuid)); return 0; }
inline int cudaRuntimeGetVersion(int* version) { *version = 13000; return 0; }
inline int cudaDriverGetVersion(int* version) { *version = 13000; return 0; }
inline int cudaMalloc(void** ptr, size_t bytes) { return posix_memalign(ptr, 65536, bytes); }
inline int cudaFree(void* ptr) { std::free(ptr); return 0; }
inline int cudaHostRegister(void*, size_t, int) { return 0; }
inline int cudaHostUnregister(void*) { return 0; }
inline int cudaPointerGetAttributes(cudaPointerAttributes* a, void*) { a->type=2; return 0; }
inline int cudaMemcpy(void* dst, const void* src, size_t bytes, int kind) {
    std::memcpy(dst, src, bytes);
    if (kind == cudaMemcpyDeviceToHost && std::getenv("G3_MOCK_CORRUPT")) static_cast<char*>(dst)[0] ^= 1;
    return 0;
}
inline int cudaMemset(void* ptr, int value, size_t bytes) { std::memset(ptr,value,bytes); return 0; }
inline int cudaDeviceSynchronize() { return 0; }
struct CUfileError_t { int err; };
constexpr int CU_FILE_SUCCESS=0, CU_FILE_DEVICE_NOT_SUPPORTED=5009;
constexpr int CU_FILE_IO_NOT_SUPPORTED=5008, CU_FILE_PLATFORM_NOT_SUPPORTED=5007;
constexpr int CU_FILE_ALLOW_COMPAT_MODE=1, CU_FILE_HANDLE_TYPE_OPAQUE_FD=1;
using CUfileHandle_t = int;
struct CUfileDescr_t { int type; struct { int fd; } handle; };
struct CUfileDrvProps_t { struct { unsigned dcontrolflags; } nvfs; };
inline CUfileError_t cuFileDriverOpen() { return {0}; }
inline CUfileError_t cuFileDriverClose() { return {0}; }
inline CUfileError_t cuFileGetVersion(int* v) { *v=1160; return {0}; }
inline CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t* p) {
    p->nvfs.dcontrolflags=std::getenv("G3_MOCK_COMPAT") ? 2 : 0; return {0};
}
inline CUfileError_t cuFileHandleRegister(int* f, CUfileDescr_t* d) {
    if (std::getenv("G3_MOCK_UNSUPPORTED")) return {CU_FILE_IO_NOT_SUPPORTED};
    *f=d->handle.fd; return {0};
}
inline void cuFileHandleDeregister(int) {}
inline CUfileError_t cuFileBufRegister(void*,size_t,int) { return {0}; }
inline CUfileError_t cuFileBufDeregister(void*) { return {0}; }
inline uint64_t mock_read_bytes=0;
inline ssize_t cuFileRead(int fd, void* ptr, size_t n, off_t off, off_t) {
    ssize_t result=pread(fd, ptr, n, off); if (result>0) mock_read_bytes+=result; return result;
}
struct CUfilePerGpuStats_t {
    char uuid[64];
    uint64_t read_bytes,n_total_reads,n_p2p_reads,n_nvfs_reads,n_posix_reads;
    uint64_t n_unaligned_reads,n_dr_reads,n_reads_err;
};
struct CUfileStatsLevel3_t { unsigned num_gpus; CUfilePerGpuStats_t per_gpu_stats[16]; };
inline CUfileError_t cuFileSetStatsLevel(int) { return {0}; }
inline CUfileError_t cuFileStatsReset() { mock_read_bytes=0; return {0}; }
inline CUfileError_t cuFileStatsStart() { return {0}; }
inline CUfileError_t cuFileStatsStop() { return {0}; }
inline CUfileError_t cuFileGetStatsL3(CUfileStatsLevel3_t* s) {
    s->num_gpus=1;auto& r=s->per_gpu_stats[0];
    std::strcpy(r.uuid,"GPU-00000000-0000-0000-0000-000000000000");
    r.read_bytes=mock_read_bytes;r.n_total_reads=mock_read_bytes/65536;r.n_p2p_reads=r.n_total_reads;return {0};
}
