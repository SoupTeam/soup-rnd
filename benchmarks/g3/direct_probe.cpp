// G3 primitive: host-compiled against installed CUDA/cuFile headers, no CUDA kernels.
// G3_TEST_BACKEND is only for CPU tests and can never produce accepted hardware evidence.
#ifdef G3_TEST_BACKEND
#include "g3_test_backend.h"
#else
#include <cuda_runtime_api.h>
#include <cufile.h>
#ifdef G3_WITH_STATS
#include <cufile_stats.h>
#endif
#endif
#include <algorithm>
#include <vector>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

struct Resources {
    int input = -1;
    int output = -1;
    void* device = nullptr;
    void* host = nullptr;
    void* oracle = nullptr;
    bool pinned = false;
    bool driver = false;
    bool handle = false;
    bool buffer = false;
    CUfileHandle_t file{};
    ~Resources() {
        if (buffer) cuFileBufDeregister(device);
        if (handle) cuFileHandleDeregister(file);
        if (driver) cuFileDriverClose();
        if (device) cudaFree(device);
        if (pinned) cudaHostUnregister(host);
        std::free(host);
        std::free(oracle);
        if (input >= 0) close(input);
        if (output >= 0) close(output);
    }
};

static void cuda_check(cudaError_t status, const char* stage) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA_ERROR stage=" << stage << " code=" << int(status) << '\n';
        throw std::runtime_error(stage);
    }
}
static void file_check(CUfileError_t status, const char* stage) {
    if (status.err != CU_FILE_SUCCESS) {
        // Numeric and symbolic attribution belongs to the pinned header, not guessed errno.
        bool unsupported = status.err == CU_FILE_DEVICE_NOT_SUPPORTED ||
                           status.err == CU_FILE_IO_NOT_SUPPORTED ||
                           status.err == CU_FILE_PLATFORM_NOT_SUPPORTED;
        std::cout << "{\"event\":\"CUFILE_ERROR\",\"stage\":\"" << stage
                  << "\",\"code\":" << int(status.err)
                  << ",\"explicit_not_supported\":" << (unsupported ? "true" : "false")
                  << "}\n";
        throw std::runtime_error(stage);
    }
}
static std::string gpu_uuid() {
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, 0), "device_uuid");
    const cudaUUID_t& uuid = properties.uuid;
    char result[41];
    char* cursor = result;
    cursor += std::snprintf(cursor, sizeof(result), "GPU-");
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) *cursor++ = '-';
        cursor += std::snprintf(cursor, sizeof(result) - size_t(cursor - result), "%02x",
                                static_cast<unsigned char>(uuid.bytes[index]));
    }
    return result;
}
static double cpu_seconds() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage");
    return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 +
           usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
}
static std::string kernel_host_metric(const char* field) {
    std::ifstream source("/proc/self/status");
    std::string line;
    while (std::getline(source, line)) {
        std::istringstream parts(line);
        std::string name, unit;
        unsigned long long value;
        if (parts >> name >> value >> unit) {
            if (name == std::string(field) + ":" && unit == "kB")
                return std::to_string(value * 1024ULL);
        }
    }
    return "null";
}
static void full_write(int fd, const void* data, size_t bytes) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t result = write(fd, static_cast<const char*>(data) + done, bytes - done);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) throw std::runtime_error("readback_write");
        done += static_cast<size_t>(result);
    }
}
int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: mode fixture bytes readback gpu_uuid stats iterations");
        std::string mode = argv[1];
        if (mode != "control" && mode != "direct") throw std::runtime_error("invalid_mode");
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(argv[3], &end, 10);
        if (!end || *end || parsed < 65536 || parsed > 67108864 || parsed % 65536)
            throw std::runtime_error("aligned_size_64KiB_to_64MiB_required");
        size_t bytes = static_cast<size_t>(parsed);
        if (std::string(argv[6]) != "on" && std::string(argv[6]) != "off")
            throw std::runtime_error("invalid_stats_mode");
        bool want_stats = std::string(argv[6]) == "on";
        std::string iterations_arg = argv[7];
        if (iterations_arg != "1" && iterations_arg != "5")
            throw std::runtime_error("iterations_must_be_1_or_5");
        int iterations = iterations_arg == "1" ? 1 : 5;
        Resources resource;
        resource.input = open(argv[2], O_RDONLY | O_DIRECT | O_NOFOLLOW);
        if (resource.input < 0) throw std::runtime_error("open_odirect");
        struct stat info{};
        if (fstat(resource.input, &info) || !S_ISREG(info.st_mode) || info.st_size != off_t(bytes))
            throw std::runtime_error("fixture_must_be_regular_exact_size");
        cuda_check(cudaSetDevice(0), "set_device");
        std::string uuid = gpu_uuid();
        if (uuid != argv[5]) throw std::runtime_error("gpu_uuid_mismatch");
        int runtime_version = 0, driver_version = 0;
        cuda_check(cudaRuntimeGetVersion(&runtime_version), "runtime_version");
        cuda_check(cudaDriverGetVersion(&driver_version), "driver_version");
        cuda_check(cudaMalloc(&resource.device, bytes), "device_alloc");
        cudaPointerAttributes attributes{};
        cuda_check(cudaPointerGetAttributes(&attributes, resource.device), "pointer_type");
        if (attributes.type != cudaMemoryTypeDevice) throw std::runtime_error("not_device_memory");
        if (posix_memalign(&resource.host, 65536, bytes)) throw std::runtime_error("host_alloc");
        if (mode == "control") {
            cuda_check(cudaHostRegister(resource.host, bytes, cudaHostRegisterDefault), "host_pin");
            resource.pinned = true;
        }
        if (posix_memalign(&resource.oracle, 65536, bytes))
            throw std::runtime_error("oracle_alloc");
        // Independent uninstrumented source bytes. The Python driver also checks full SHA256.
        ssize_t oracle_bytes = pread(resource.input, resource.oracle, bytes, 0);
        if (oracle_bytes != ssize_t(bytes)) throw std::runtime_error("oracle_short_read");
        int cufile_version = 0;
        bool compat_disabled = false;
        bool stats_available = false;
        uint64_t route_bytes = 0, p2p_reads = 0, nvfs_reads = 0, posix_reads = 0;
        uint64_t unaligned_reads = 0, dr_reads = 0, read_errors = 0, total_reads = 0;
        if (mode == "direct") {
            file_check(cuFileDriverOpen(), "driver_open");
            resource.driver = true;
            file_check(cuFileGetVersion(&cufile_version), "cufile_version");
            CUfileDrvProps_t properties{};
            file_check(cuFileDriverGetProperties(&properties), "driver_properties");
            compat_disabled = !(properties.nvfs.dcontrolflags & (1u << CU_FILE_ALLOW_COMPAT_MODE));
            if (!compat_disabled) throw std::runtime_error("compatibility_mode_not_disabled");
            CUfileDescr_t descriptor{};
            descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            descriptor.handle.fd = resource.input;
            file_check(cuFileHandleRegister(&resource.file, &descriptor), "handle_register");
            resource.handle = true;
            file_check(cuFileBufRegister(resource.device, bytes, 0), "buffer_register");
            resource.buffer = true;
#ifdef G3_WITH_STATS
            if (want_stats) {
                file_check(cuFileSetStatsLevel(3), "stats_level");
                file_check(cuFileStatsReset(), "stats_reset");
                file_check(cuFileStatsStart(), "stats_start");
                stats_available = true;
            }
#else
            if (want_stats) throw std::runtime_error("stats_api_not_compiled");
#endif
        }
        std::string vm_lck_before = kernel_host_metric("VmLck");
        std::string vm_pin_before = kernel_host_metric("VmPin");
        std::vector<double> elapsed_samples, cpu_samples;
        for (int iteration = 0; iteration < iterations; ++iteration) {
        cuda_check(cudaMemset(resource.device, 0xA5, bytes), "poison_destination");
        cuda_check(cudaDeviceSynchronize(), "before_read_sync");
        double cpu_start = cpu_seconds();
        auto started = std::chrono::steady_clock::now();
        if (mode == "control") {
            size_t done = 0;
            while (done < bytes) {
                ssize_t result = pread(resource.input,
                    static_cast<char*>(resource.host) + done, bytes - done, off_t(done));
                if (result < 0 && errno == EINTR) continue;
                if (result <= 0) throw std::runtime_error("short_odirect_read");
                done += static_cast<size_t>(result);
                if (done != bytes && done % 4096) throw std::runtime_error("unaligned_short_read");
            }
            cuda_check(cudaMemcpy(resource.device, resource.host, bytes, cudaMemcpyHostToDevice),
                       "host_to_device");
        } else {
            errno = 0;
            ssize_t read_bytes = cuFileRead(resource.file, resource.device, bytes, 0, 0);
            if (read_bytes != ssize_t(bytes)) {
                std::cout << "{\"event\":\"CUFILE_READ_ERROR\",\"result\":" << read_bytes
                          << ",\"errno\":" << errno << "}\n";
                throw std::runtime_error("short_or_failed_cufile_read");
            }
        }
        cuda_check(cudaDeviceSynchronize(), "completed_read_sync");
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        double cpu_elapsed = cpu_seconds() - cpu_start;
        cuda_check(cudaMemcpy(resource.host, resource.device, bytes, cudaMemcpyDeviceToHost),
                   "per_iteration_readback");
        if (std::memcmp(resource.host, resource.oracle, bytes))
            throw std::runtime_error("per_iteration_integrity");
        if (iterations == 1 || iteration >= 2) {
            elapsed_samples.push_back(elapsed); cpu_samples.push_back(cpu_elapsed);
        }
        }
        auto sorted_elapsed = elapsed_samples;
        std::sort(sorted_elapsed.begin(), sorted_elapsed.end());
        std::sort(cpu_samples.begin(), cpu_samples.end());
        double elapsed = sorted_elapsed[sorted_elapsed.size()/2];
        double cpu_elapsed = cpu_samples[cpu_samples.size()/2];
#ifdef G3_WITH_STATS
        if (stats_available) {
            file_check(cuFileStatsStop(), "stats_stop");
            CUfileStatsLevel3_t stats{};
            file_check(cuFileGetStatsL3(&stats), "stats_readback");
            bool found = false;
            if (stats.num_gpus > 16) throw std::runtime_error("stats_gpu_count");
            for (unsigned index = 0; index < stats.num_gpus; ++index) {
                const auto& row = stats.per_gpu_stats[index];
                if (std::strncmp(row.uuid, uuid.c_str(), sizeof(row.uuid)) != 0) continue;
                if (found) throw std::runtime_error("duplicate_stats_gpu_uuid");
                found = true;
                route_bytes = row.read_bytes; total_reads = row.n_total_reads;
                p2p_reads = row.n_p2p_reads; nvfs_reads = row.n_nvfs_reads;
                posix_reads = row.n_posix_reads; unaligned_reads = row.n_unaligned_reads;
                dr_reads = row.n_dr_reads; read_errors = row.n_reads_err;
            }
            if (!found) throw std::runtime_error("target_gpu_absent_from_stats");
        }
#endif
        // Readback and verification are outside the bracket and cannot make a bad read faster.
        cuda_check(cudaMemcpy(resource.host, resource.device, bytes, cudaMemcpyDeviceToHost),
                   "device_to_host_readback");
        resource.output = open(argv[4], O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
        if (resource.output < 0) throw std::runtime_error("exclusive_readback_output");
        full_write(resource.output, resource.host, bytes);
        if (fsync(resource.output)) throw std::runtime_error("readback_fsync");
#ifdef G3_TEST_BACKEND
        const char* backend = "SYNTHETIC_CPU_MOCK";
#else
        const char* backend = "NATIVE_CUDA_CUFILE";
#endif
        std::cout << "{\"event\":\"READ_COMPLETE\",\"backend\":\"" << backend
          << "\",\"mode\":\"" << mode << "\",\"gpu_uuid\":\"" << uuid
          << "\",\"bytes\":" << bytes << ",\"elapsed_s\":" << elapsed
          << ",\"read_operations\":" << iterations
          << ",\"timed_samples\":" << elapsed_samples.size()
          << ",\"kernel_vm_lck_after_alloc\":" << vm_lck_before
          << ",\"kernel_vm_pin_after_alloc\":" << vm_pin_before
          << ",\"kernel_vm_lck_after_read\":" << kernel_host_metric("VmLck")
          << ",\"kernel_vm_pin_after_read\":" << kernel_host_metric("VmPin")
          << ",\"cpu_s\":" << cpu_elapsed << ",\"explicit_pinned_host_bytes\":"
          << (resource.pinned ? bytes : 0) << ",\"total_pinned_host_bytes\":null,\"cuda_runtime\":" << runtime_version
          << ",\"cuda_driver\":" << driver_version << ",\"cufile_version\":"
          << cufile_version << ",\"device_memory\":true,\"compat_disabled\":"
          << (compat_disabled ? "true" : "false") << ",\"stats_available\":"
          << (stats_available ? "true" : "false") << ",\"route_read_bytes\":" << route_bytes
          << ",\"total_reads\":" << total_reads << ",\"p2p_reads\":" << p2p_reads
          << ",\"nvfs_reads\":" << nvfs_reads << ",\"posix_reads\":" << posix_reads
          << ",\"unaligned_reads\":" << unaligned_reads << ",\"dr_reads\":" << dr_reads
          << ",\"read_errors\":" << read_errors << ",\"elapsed_samples_s\":[";
        for (size_t index=0; index<elapsed_samples.size(); ++index) {
            if (index) std::cout << ",";
            std::cout << elapsed_samples[index];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "G3_FAILURE " << error.what() << " errno=" << errno << '\n';
        return 2;
    }
}
