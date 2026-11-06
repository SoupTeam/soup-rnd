// G3 Windows DirectStorage probe (PROTOCOL.txt in this folder).
//
//   ds_probe selftest <dir>                  small end-to-end check of both arms, JSON to stdout
//   ds_probe prepare  <fixture>              write and read back the 1 GiB fixture
//   ds_probe run <A|B> <fixture> <out.json>  one run: 2 warm-up + 3 timed reads, oracle, F1 watch
//
// A: unbuffered ReadFile into two pinned host buffers, cuMemcpyHtoDAsync into a cuMemAlloc buffer.
// B: DirectStorage file queue into a shared D3D12 buffer imported into CUDA.
// CUDA is used through the driver API (nvcuda.dll); dstorage.dll is loaded from the exe folder.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#include <psapi.h>
#include <shellapi.h>
#include <fcntl.h>
#include <io.h>
#include <tlhelp32.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "dstorage_min.h"

static const uint64_t FIXTURE_BYTES = 1ull << 30;
static const uint32_t CHUNK = 32u << 20;
static const uint64_t SEED = 20261104ull;
static const int WARMUP = 2, TIMED = 3;
static const uint64_t FOREIGN_LIMIT = 1000000000ull;

// ------------------------------------------------------------------------------------- output

static std::string g_error;
static bool fail(const char *what, long long code) {
    if (code < 0 && code >= -0x80000000LL) code &= 0xffffffffLL;  // HRESULT as 32 bits
    if (g_error.empty()) {
        char buf[256];
        snprintf(buf, sizeof buf, "%s (0x%llx)", what, (unsigned long long)code);
        g_error = buf;
    }
    return false;
}
static std::string esc(const std::string &s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o;
}
static std::string narrow(const wchar_t *w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}
static double now_s() {
    static LARGE_INTEGER f = {};
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

// ------------------------------------------------------------------------------------ fixture

static inline uint64_t splitmix(uint64_t &x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static void fill(uint8_t *dst, uint64_t offset, size_t bytes) {  // offset and bytes: 4 KiB multiples
    for (size_t done = 0; done < bytes; done += 4096) {
        uint64_t state = SEED * 0x100000001B3ull ^ ((offset + done) >> 12);
        uint64_t *w = (uint64_t *)(dst + done);
        for (int i = 0; i < 512; i++) w[i] = splitmix(state);
    }
}
// Returns the first differing absolute offset, or UINT64_MAX when equal.
static uint64_t compare(const uint8_t *got, uint64_t offset, size_t bytes, std::vector<uint8_t> &tmp) {
    tmp.resize(bytes);
    fill(tmp.data(), offset, bytes);
    if (memcmp(got, tmp.data(), bytes) == 0) return UINT64_MAX;
    for (size_t i = 0; i < bytes; i++)
        if (got[i] != tmp[i]) return offset + i;
    return UINT64_MAX;
}

// ------------------------------------------------------------------------------ CUDA (driver)

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUextMemory_st *CUexternalMemory;
typedef unsigned long long CUdeviceptr;
struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    int type;
    union {
        int fd;
        struct { void *handle; const void *name; } win32;
        const void *nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    unsigned long long offset, size;
    unsigned int flags;
    unsigned int reserved[16];
};
static const int CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5;
static const unsigned CUDA_EXTERNAL_MEMORY_DEDICATED = 1;

struct Cuda {
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceGetLuid)(char *, unsigned *, CUdevice);
    CUresult (*DriverGetVersion)(int *);
    CUresult (*PrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*CtxSynchronize)();
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemHostAlloc)(void **, size_t, unsigned);
    CUresult (*MemFreeHost)(void *);
    CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*EventCreate)(CUevent *, unsigned);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventSynchronize)(CUevent);
    CUresult (*ImportExternalMemory)(CUexternalMemory *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
    CUresult (*ExternalMemoryGetMappedBuffer)(CUdeviceptr *, CUexternalMemory,
                                              const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
    CUresult (*DestroyExternalMemory)(CUexternalMemory);
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    char name[256] = {};
    unsigned char luid[8] = {};
    int driver = 0;
};
static Cuda cu;

template <typename T> static bool sym(HMODULE m, const char *name, T &out) {
    out = (T)(void *)GetProcAddress(m, name);
    return out ? true : fail(name, 0);
}
#define CU(call) do { CUresult r_ = (call); if (r_) return fail(#call, r_); } while (0)

static bool cuda_init() {
    HMODULE m = LoadLibraryW(L"nvcuda.dll");
    if (!m) return fail("LoadLibrary nvcuda.dll", GetLastError());
    bool ok = sym(m, "cuInit", cu.Init) && sym(m, "cuDeviceGet", cu.DeviceGet) &&
              sym(m, "cuDeviceGetName", cu.DeviceGetName) && sym(m, "cuDeviceGetLuid", cu.DeviceGetLuid) &&
              sym(m, "cuDriverGetVersion", cu.DriverGetVersion) &&
              sym(m, "cuDevicePrimaryCtxRetain", cu.PrimaryCtxRetain) &&
              sym(m, "cuCtxSetCurrent", cu.CtxSetCurrent) && sym(m, "cuCtxSynchronize", cu.CtxSynchronize) &&
              sym(m, "cuMemAlloc_v2", cu.MemAlloc) && sym(m, "cuMemFree_v2", cu.MemFree) &&
              sym(m, "cuMemHostAlloc", cu.MemHostAlloc) && sym(m, "cuMemFreeHost", cu.MemFreeHost) &&
              sym(m, "cuMemcpyHtoDAsync_v2", cu.MemcpyHtoDAsync) && sym(m, "cuMemcpyDtoH_v2", cu.MemcpyDtoH) &&
              sym(m, "cuMemcpyDtoD_v2", cu.MemcpyDtoD) && sym(m, "cuStreamCreate", cu.StreamCreate) &&
              sym(m, "cuStreamSynchronize", cu.StreamSynchronize) && sym(m, "cuEventCreate", cu.EventCreate) &&
              sym(m, "cuEventRecord", cu.EventRecord) && sym(m, "cuEventSynchronize", cu.EventSynchronize) &&
              sym(m, "cuImportExternalMemory", cu.ImportExternalMemory) &&
              sym(m, "cuExternalMemoryGetMappedBuffer", cu.ExternalMemoryGetMappedBuffer) &&
              sym(m, "cuDestroyExternalMemory", cu.DestroyExternalMemory);
    if (!ok) return false;
    CU(cu.Init(0));
    CU(cu.DriverGetVersion(&cu.driver));
    CU(cu.DeviceGet(&cu.dev, 0));
    CU(cu.DeviceGetName(cu.name, sizeof cu.name - 1, cu.dev));
    unsigned mask = 0;
    CU(cu.DeviceGetLuid((char *)cu.luid, &mask, cu.dev));
    CU(cu.PrimaryCtxRetain(&cu.ctx, cu.dev));
    CU(cu.CtxSetCurrent(cu.ctx));
    return true;
}

// ---------------------------------------------------------------------------------- F1 watch

struct ProcRead { std::string name; uint64_t created; uint64_t bytes; };
static void process_reads(std::map<DWORD, ProcRead> &out, int &uninspectable) {
    out.clear();
    uninspectable = 0;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) { uninspectable = -1; return; }
    PROCESSENTRY32W e = {};
    e.dwSize = sizeof e;
    DWORD self = GetCurrentProcessId();
    for (BOOL ok = Process32FirstW(snap, &e); ok; ok = Process32NextW(snap, &e)) {
        if (e.th32ProcessID == self || e.th32ProcessID == 0) continue;
        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, e.th32ProcessID);
        IO_COUNTERS io = {};
        FILETIME c, x, k, u;
        if (!h || !GetProcessIoCounters(h, &io) || !GetProcessTimes(h, &c, &x, &k, &u)) {
            uninspectable++;
            if (h) CloseHandle(h);
            continue;
        }
        out[e.th32ProcessID] = {narrow(e.szExeFile),
                                ((uint64_t)c.dwHighDateTime << 32) | c.dwLowDateTime, io.ReadTransferCount};
        CloseHandle(h);
    }
    CloseHandle(snap);
}

struct Watch {
    HANDLE stop = nullptr, thread = nullptr;
    int samples = 0, off_mains = 0, unknown_power = 0, no_battery = 0;
    double max_gap_s = 0;
    std::map<DWORD, ProcRead> start_reads;
    int uninspectable_start = 0;
    void sample(uint64_t &last) {
        SYSTEM_POWER_STATUS p;
        if (!GetSystemPowerStatus(&p)) unknown_power++;
        else if (p.BatteryFlag == 128) no_battery++;
        else if (p.ACLineStatus == 0) off_mains++;
        else if (p.ACLineStatus != 1) unknown_power++;
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
        if (last) max_gap_s = std::max(max_gap_s, (double)(t - last) / 1e7);
        last = t;
        samples++;
    }
    static DWORD WINAPI loop(void *self) {
        Watch *w = (Watch *)self;
        uint64_t last = 0;
        do { w->sample(last); } while (WaitForSingleObject(w->stop, 2000) == WAIT_TIMEOUT);
        w->sample(last);
        return 0;
    }
    void begin() {
        process_reads(start_reads, uninspectable_start);
        stop = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        thread = CreateThread(nullptr, 0, loop, this, 0, nullptr);
    }
    std::string end() {
        SetEvent(stop);
        WaitForSingleObject(thread, INFINITE);
        std::map<DWORD, ProcRead> after;
        int uninspectable_end = 0;
        process_reads(after, uninspectable_end);
        uint64_t worst = 0;
        std::string worst_name;
        for (auto &kv : after) {
            auto it = start_reads.find(kv.first);
            uint64_t before = (it != start_reads.end() && it->second.created == kv.second.created)
                                  ? it->second.bytes : 0;
            uint64_t delta = kv.second.bytes >= before ? kv.second.bytes - before : 0;
            if (delta > worst) { worst = delta; worst_name = kv.second.name; }
        }
        const char *power = off_mains ? "void" : (unknown_power ? "unknown" : "ok");
        const char *suspend = max_gap_s > 30.0 ? "void" : "ok";
        const char *foreign = (uninspectable_start < 0) ? "unknown" : (worst >= FOREIGN_LIMIT ? "void" : "ok");
        const char *outcome = (!strcmp(power, "void") || !strcmp(suspend, "void") || !strcmp(foreign, "void"))
                                  ? "void"
                                  : ((!strcmp(power, "unknown") || !strcmp(foreign, "unknown")) ? "unknown" : "ok");
        char buf[1024];
        snprintf(buf, sizeof buf,
                 "{\"outcome\": \"%s\", \"power\": {\"outcome\": \"%s\", \"samples\": %d, \"off_mains\": %d,"
                 " \"unknown\": %d, \"no_battery\": %d}, \"suspend\": {\"outcome\": \"%s\", \"max_gap_s\": %.3f},"
                 " \"foreign_reads\": {\"outcome\": \"%s\", \"max_other_process_bytes\": %llu,"
                 " \"max_other_process\": \"%s\", \"uninspectable_start\": %d, \"uninspectable_end\": %d,"
                 " \"limit_bytes\": %llu}}",
                 outcome, power, samples, off_mains, unknown_power, no_battery, suspend, max_gap_s, foreign,
                 (unsigned long long)worst, esc(worst_name).c_str(), uninspectable_start, uninspectable_end,
                 (unsigned long long)FOREIGN_LIMIT);
        return buf;
    }
};

// ------------------------------------------------------------------------------- process facts

struct Mem { uint64_t priv, ws, peak_ws; };
static Mem mem() {
    PROCESS_MEMORY_COUNTERS_EX m = {};
    m.cb = sizeof m;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&m, sizeof m);
    return {m.PrivateUsage, m.WorkingSetSize, m.PeakWorkingSetSize};
}
static double cpu_s() {
    FILETIME c, x, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &x, &k, &u);
    auto v = [](FILETIME f) { return (double)(((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e7; };
    return v(k) + v(u);
}
static std::string mem_json(const Mem &m) {
    char b[160];
    snprintf(b, sizeof b, "{\"private_bytes\": %llu, \"working_set\": %llu, \"peak_working_set\": %llu}",
             (unsigned long long)m.priv, (unsigned long long)m.ws, (unsigned long long)m.peak_ws);
    return b;
}

// ------------------------------------------------------------------------------------- arm A

struct ArmA {
    HANDLE file = INVALID_HANDLE_VALUE;
    void *host[2] = {};
    CUdeviceptr dst = 0;
    CUstream stream = nullptr;
    CUevent done[2] = {};
    uint64_t bytes = 0;
    bool setup(const wchar_t *path, uint64_t size) {
        bytes = size;
        file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (file == INVALID_HANDLE_VALUE) return fail("CreateFileW unbuffered", GetLastError());
        for (int i = 0; i < 2; i++) {
            CU(cu.MemHostAlloc(&host[i], CHUNK, 0));
            CU(cu.EventCreate(&done[i], 0));
        }
        CU(cu.StreamCreate(&stream, 0));
        CU(cu.MemAlloc(&dst, size));
        return true;
    }
    bool read_once(double &seconds) {
        double t0 = now_s();
        uint64_t chunks = bytes / CHUNK;
        for (uint64_t i = 0; i < chunks; i++) {
            int b = (int)(i % 2);
            if (i >= 2) CU(cu.EventSynchronize(done[b]));
            OVERLAPPED ov = {};
            uint64_t off = i * CHUNK;
            ov.Offset = (DWORD)off;
            ov.OffsetHigh = (DWORD)(off >> 32);
            DWORD got = 0;
            if (!ReadFile(file, host[b], CHUNK, &got, &ov)) return fail("ReadFile", GetLastError());
            if (got != CHUNK) return fail("short ReadFile", got);
            CU(cu.MemcpyHtoDAsync(dst + off, host[b], CHUNK, stream));
            CU(cu.EventRecord(done[b], stream));
        }
        CU(cu.StreamSynchronize(stream));
        seconds = now_s() - t0;
        return true;
    }
};

// ------------------------------------------------------------------------------------- arm B

struct ArmB {
    IDXGIAdapter1 *adapter = nullptr;
    ID3D12Device *device = nullptr;
    ID3D12Resource *buffer = nullptr;
    ID3D12Fence *fence = nullptr;
    HANDLE fence_event = nullptr, shared = nullptr;
    IDStorageFactory *factory = nullptr;
    IDStorageFile *file = nullptr;
    IDStorageQueue *queue = nullptr;
    CUexternalMemory ext = nullptr;
    CUdeviceptr mapped = 0;
    uint64_t bytes = 0, fence_value = 0;
    std::string adapter_name;
    bool setup(const wchar_t *path, uint64_t size) {
        bytes = size;
        IDXGIFactory1 *dxgi = nullptr;
        HRESULT hr = CreateDXGIFactory1(IID_IDXGIFactory1, (void **)&dxgi);
        if (FAILED(hr)) return fail("CreateDXGIFactory1", hr);
        for (UINT i = 0; dxgi->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
            DXGI_ADAPTER_DESC1 d;
            adapter->GetDesc1(&d);
            if (memcmp(&d.AdapterLuid, cu.luid, 8) == 0) { adapter_name = narrow(d.Description); break; }
            adapter->Release();
            adapter = nullptr;
        }
        dxgi->Release();
        if (!adapter) return fail("no DXGI adapter with the CUDA device LUID", 0);
        hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_ID3D12Device, (void **)&device);
        if (FAILED(hr)) return fail("D3D12CreateDevice", hr);
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
                                             nullptr, IID_ID3D12Resource, (void **)&buffer);
        if (FAILED(hr)) return fail("CreateCommittedResource shared buffer", hr);
        hr = device->CreateSharedHandle(buffer, nullptr, GENERIC_ALL, nullptr, &shared);
        if (FAILED(hr)) return fail("CreateSharedHandle", hr);
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC h = {};
        h.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        h.handle.win32.handle = shared;
        h.size = size;  // a buffer's allocation size is its width rounded to 64 KiB; 1 GiB is aligned
        h.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        CU(cu.ImportExternalMemory(&ext, &h));
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd = {};
        bd.size = size;
        CU(cu.ExternalMemoryGetMappedBuffer(&mapped, ext, &bd));
        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, (void **)&fence);
        if (FAILED(hr)) return fail("CreateFence", hr);
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        wchar_t dll[MAX_PATH];
        GetModuleFileNameW(nullptr, dll, MAX_PATH);
        wchar_t *slash = wcsrchr(dll, L'\\');
        if (slash) wcscpy(slash + 1, L"dstorage.dll");
        HMODULE ds = LoadLibraryW(dll);
        if (!ds) return fail("LoadLibrary dstorage.dll next to ds_probe.exe", GetLastError());
        PFN_DStorageGetFactory get = (PFN_DStorageGetFactory)(void *)GetProcAddress(ds, "DStorageGetFactory");
        if (!get) return fail("DStorageGetFactory missing", 0);
        hr = get(IID_IDStorageFactory_, (void **)&factory);
        if (FAILED(hr)) return fail("DStorageGetFactory", hr);
        hr = factory->OpenFile(path, IID_IDStorageFile_, (void **)&file);
        if (FAILED(hr)) return fail("IDStorageFactory::OpenFile", hr);
        DSTORAGE_QUEUE_DESC q = {};
        q.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        q.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
        q.Priority = DSTORAGE_PRIORITY_NORMAL;
        q.Name = "g3";
        q.Device = device;
        hr = factory->CreateQueue(&q, IID_IDStorageQueue_, (void **)&queue);
        if (FAILED(hr)) return fail("IDStorageFactory::CreateQueue", hr);
        return true;
    }
    bool read_once(double &seconds) {
        double t0 = now_s();
        uint64_t chunks = bytes / CHUNK;
        for (uint64_t i = 0; i < chunks; i++) {
            DSTORAGE_REQUEST r = {};
            r.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            r.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            r.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            r.Source.File.Source = file;
            r.Source.File.Offset = i * CHUNK;
            r.Source.File.Size = CHUNK;
            r.Destination.Buffer.Resource = buffer;
            r.Destination.Buffer.Offset = i * CHUNK;
            r.Destination.Buffer.Size = CHUNK;
            r.UncompressedSize = CHUNK;
            queue->EnqueueRequest(&r);
        }
        fence_value++;
        queue->EnqueueSignal(fence, fence_value);
        queue->Submit();
        HRESULT hr = fence->SetEventOnCompletion(fence_value, fence_event);
        if (FAILED(hr)) return fail("SetEventOnCompletion", hr);
        if (WaitForSingleObject(fence_event, 120000) != WAIT_OBJECT_0) return fail("DirectStorage fence timeout", 0);
        seconds = now_s() - t0;
        DSTORAGE_ERROR_RECORD rec = {};
        queue->RetrieveErrorRecord(&rec);
        if (rec.FailureCount) return fail("DirectStorage request failed", rec.FirstFailure.HResult);
        if (fence->GetCompletedValue() < fence_value) return fail("fence not completed", 0);
        return true;
    }
};

// ------------------------------------------------------------------------------- byte oracle

// Copies the device buffer back outside any timed interval and checks every byte, then checks that
// a one-byte corruption of the copied data is detected.
static bool verify(CUdeviceptr src, uint64_t bytes, std::string &json) {
    std::vector<uint8_t> host(CHUNK), tmp;
    uint64_t first_bad = UINT64_MAX;
    for (uint64_t off = 0; off < bytes && first_bad == UINT64_MAX; off += CHUNK) {
        CU(cu.MemcpyDtoH(host.data(), src + off, CHUNK));
        first_bad = compare(host.data(), off, CHUNK, tmp);
    }
    CU(cu.MemcpyDtoH(host.data(), src, CHUNK));
    host[12345] ^= 0x01;
    bool control = compare(host.data(), 0, CHUNK, tmp) == 12345;
    char b[200];
    snprintf(b, sizeof b, "{\"all_bytes_equal\": %s, \"first_mismatch\": %lld, \"corruption_control_detected\": %s}",
             first_bad == UINT64_MAX ? "true" : "false", first_bad == UINT64_MAX ? -1LL : (long long)first_bad,
             control ? "true" : "false");
    json = b;
    if (first_bad != UINT64_MAX) return fail("byte oracle mismatch", (long long)first_bad);
    if (!control) return fail("corruption control not detected", 0);
    return true;
}

// ------------------------------------------------------------------------------------- modes

static bool prepare(const wchar_t *path, uint64_t size) {
    HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return fail("CreateFileW fixture (must not exist)", GetLastError());
    std::vector<uint8_t> buf(CHUNK), tmp;
    for (uint64_t off = 0; off < size; off += CHUNK) {
        fill(buf.data(), off, CHUNK);
        DWORD put = 0;
        if (!WriteFile(h, buf.data(), CHUNK, &put, nullptr) || put != CHUNK) return fail("WriteFile", GetLastError());
    }
    if (!FlushFileBuffers(h)) return fail("FlushFileBuffers", GetLastError());
    CloseHandle(h);
    h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) return fail("reopen fixture", GetLastError());
    for (uint64_t off = 0; off < size; off += CHUNK) {
        DWORD got = 0;
        if (!ReadFile(h, buf.data(), CHUNK, &got, nullptr) || got != CHUNK) return fail("readback", GetLastError());
        uint64_t bad = compare(buf.data(), off, CHUNK, tmp);
        if (bad != UINT64_MAX) return fail("fixture readback mismatch", (long long)bad);
    }
    CloseHandle(h);
    return true;
}

static std::string median_and_list(std::vector<double> v, double &med) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) { char b[32]; snprintf(b, sizeof b, "%s%.6f", i ? ", " : "", v[i]); s += b; }
    s += "]";
    std::sort(v.begin(), v.end());
    med = v[v.size() / 2];
    return s;
}

static int run(char arm, const wchar_t *path, uint64_t size, int warmup, int timed, FILE *out) {
    Watch watch;
    watch.begin();
    Mem m0 = mem();
    double setup0 = now_s();
    ArmA a;
    ArmB b;
    bool ok = cuda_init() && (arm == 'A' ? a.setup(path, size) : b.setup(path, size));
    double setup_s = now_s() - setup0;
    Mem m1 = mem();
    std::vector<double> times;
    double cpu0 = 0, cpu1 = 0, dtod_s = -1;
    for (int i = 0; ok && i < warmup + timed; i++) {
        if (i == warmup) cpu0 = cpu_s();
        double s = 0;
        ok = arm == 'A' ? a.read_once(s) : b.read_once(s);
        if (ok && i >= warmup) times.push_back(s);
    }
    cpu1 = cpu_s();
    Mem m2 = mem();
    std::string oracle = "null";
    CUdeviceptr dst = arm == 'A' ? a.dst : b.mapped;
    if (ok) ok = verify(dst, size, oracle);
    if (ok && arm == 'B') {  // reported, not the arm's number
        CUdeviceptr own = 0;
        if (cu.MemAlloc(&own, size) == 0) {
            double t0 = now_s();
            if (cu.MemcpyDtoD(own, b.mapped, size) == 0 && cu.CtxSynchronize() == 0) dtod_s = now_s() - t0;
            cu.MemFree(own);
        }
    }
    std::string validity = watch.end();
    double med = 0;
    std::string list = times.empty() ? "[]" : median_and_list(times, med);
    fprintf(out,
            "{\"schema\": \"g3-ds-run-v1\", \"arm\": \"%c\", \"ok\": %s, \"error\": \"%s\", \"bytes_per_read\": %llu,"
            " \"chunk_bytes\": %u, \"warmup\": %d, \"timed\": %d, \"read_seconds\": %s, \"median_seconds\": %.6f,"
            " \"useful_gb_per_s\": %.6f, \"cpu_seconds_per_read\": %.6f, \"setup_seconds\": %.6f,"
            " \"dtod_copy_seconds\": %.6f, \"explicit_pinned_host_bytes\": %u, \"oracle\": %s,"
            " \"memory\": {\"start\": %s, \"after_setup\": %s, \"after_reads\": %s},"
            " \"cuda\": {\"device\": \"%s\", \"driver_version\": %d}, \"d3d12_adapter\": \"%s\", \"validity\": %s}\n",
            arm, ok ? "true" : "false", esc(g_error).c_str(), (unsigned long long)size, CHUNK, warmup, timed,
            list.c_str(), med, med > 0 ? (double)size / med / 1e9 : 0.0,
            times.empty() ? 0.0 : (cpu1 - cpu0) / (double)times.size(), setup_s, dtod_s,
            arm == 'A' ? 2 * CHUNK : 0u, oracle.c_str(), mem_json(m0).c_str(), mem_json(m1).c_str(),
            mem_json(m2).c_str(), esc(cu.name).c_str(), cu.driver, esc(b.adapter_name).c_str(), validity.c_str());
    return ok ? 0 : 1;
}

int main() {
    int argc = 0;
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argc >= 3 && !wcscmp(argv[1], L"prepare")) {
        bool ok = prepare(argv[2], FIXTURE_BYTES);
        printf("{\"mode\": \"prepare\", \"ok\": %s, \"bytes\": %llu, \"error\": \"%s\"}\n", ok ? "true" : "false",
               (unsigned long long)FIXTURE_BYTES, esc(g_error).c_str());
        return ok ? 0 : 1;
    }
    if (argc >= 5 && !wcscmp(argv[1], L"run") && (argv[2][0] == L'A' || argv[2][0] == L'B')) {
        // Never overwrite a result: CREATE_NEW fails if the file exists (msvcrt ignores "x").
        HANDLE h = CreateFileW(argv[4], GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "output exists or cannot be created\n"); return 2; }
        FILE *out = _fdopen(_open_osfhandle((intptr_t)h, 0), "w");
        if (!out) { CloseHandle(h); fprintf(stderr, "cannot open output stream\n"); return 2; }
        int code = run((char)argv[2][0], argv[3], FIXTURE_BYTES, WARMUP, TIMED, out);
        fclose(out);
        return code;
    }
    if (argc >= 3 && !wcscmp(argv[1], L"selftest")) {
        std::wstring path = std::wstring(argv[2]) + L"\\g3_selftest_64mib.bin";
        DeleteFileW(path.c_str());
        uint64_t size = 2ull * CHUNK;
        HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        std::vector<uint8_t> buf(CHUNK);
        bool ok = h != INVALID_HANDLE_VALUE;
        for (uint64_t off = 0; ok && off < size; off += CHUNK) {
            fill(buf.data(), off, CHUNK);
            DWORD put = 0;
            ok = WriteFile(h, buf.data(), CHUNK, &put, nullptr) && put == CHUNK;
        }
        if (h != INVALID_HANDLE_VALUE) { FlushFileBuffers(h); CloseHandle(h); }
        if (!ok) { printf("{\"mode\": \"selftest\", \"ok\": false, \"error\": \"cannot write selftest file\"}\n"); return 1; }
        int ca = run('A', path.c_str(), size, 0, 1, stdout);
        g_error.clear();
        int cb = run('B', path.c_str(), size, 0, 1, stdout);
        DeleteFileW(path.c_str());
        return ca || cb;
    }
    fprintf(stderr, "usage: ds_probe selftest <dir> | prepare <fixture> | run <A|B> <fixture> <out.json>\n");
    return 2;
}
