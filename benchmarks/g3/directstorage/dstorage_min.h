// Subset of dstorage.h from the Microsoft DirectStorage SDK 1.3.0 (NuGet package
// microsoft.direct3d.directstorage 1.3.0). Copyright (c) Microsoft Corporation, MIT License:
// the declarations below keep that header's names, order and layout; only the parts the G3
// probe uses are kept, plus every type that RetrieveErrorRecord writes.
#pragma once
#include <windows.h>
#include <unknwn.h>
#include <d3d12.h>

enum DSTORAGE_PRIORITY : INT8 { DSTORAGE_PRIORITY_NORMAL = 0 };
#define DSTORAGE_MAX_QUEUE_CAPACITY 0x2000

enum DSTORAGE_REQUEST_SOURCE_TYPE : UINT64 { DSTORAGE_REQUEST_SOURCE_FILE = 0 };
enum DSTORAGE_REQUEST_DESTINATION_TYPE : UINT64 { DSTORAGE_REQUEST_DESTINATION_BUFFER = 1 };

struct DSTORAGE_QUEUE_DESC {
    DSTORAGE_REQUEST_SOURCE_TYPE SourceType;
    UINT16 Capacity;
    DSTORAGE_PRIORITY Priority;
    const CHAR *Name;
    ID3D12Device *Device;
};
struct DSTORAGE_QUEUE_INFO {
    DSTORAGE_QUEUE_DESC Desc;
    UINT16 EmptySlotCount;
    UINT16 RequestCountUntilAutoSubmit;
};
enum DSTORAGE_COMPRESSION_FORMAT : UINT8 { DSTORAGE_COMPRESSION_FORMAT_NONE = 0 };
struct DSTORAGE_REQUEST_OPTIONS {
    DSTORAGE_COMPRESSION_FORMAT CompressionFormat : 8;
    UINT8 Reserved1[7];
    DSTORAGE_REQUEST_SOURCE_TYPE SourceType : 1;
    DSTORAGE_REQUEST_DESTINATION_TYPE DestinationType : 7;
    UINT64 Reserved : 48;
};

struct IDStorageFile : public IUnknown {
    virtual void STDMETHODCALLTYPE Close() = 0;
    virtual HRESULT STDMETHODCALLTYPE GetFileInformation(BY_HANDLE_FILE_INFORMATION *info) = 0;
};
struct IDStorageStatusArray;

struct DSTORAGE_SOURCE_FILE { IDStorageFile *Source; UINT64 Offset; UINT32 Size; };
struct DSTORAGE_SOURCE_MEMORY { void const *Source; UINT32 Size; };
struct DSTORAGE_DESTINATION_MEMORY { void *Buffer; UINT32 Size; };
struct DSTORAGE_DESTINATION_BUFFER { ID3D12Resource *Resource; UINT64 Offset; UINT32 Size; };
struct DSTORAGE_DESTINATION_TEXTURE_REGION {
    ID3D12Resource *Resource; UINT SubresourceIndex; D3D12_BOX Region;
};
struct DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES { ID3D12Resource *Resource; UINT FirstSubresource; };
struct DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES_RANGE {
    ID3D12Resource *Resource; UINT FirstSubresource; UINT NumSubresources;
};
struct DSTORAGE_DESTINATION_TILES {
    ID3D12Resource *Resource;
    D3D12_TILED_RESOURCE_COORDINATE TiledRegionStartCoordinate;
    D3D12_TILE_REGION_SIZE TileRegionSize;
};
union DSTORAGE_SOURCE { DSTORAGE_SOURCE_MEMORY Memory; DSTORAGE_SOURCE_FILE File; };
union DSTORAGE_DESTINATION {
    DSTORAGE_DESTINATION_MEMORY Memory;
    DSTORAGE_DESTINATION_BUFFER Buffer;
    DSTORAGE_DESTINATION_TEXTURE_REGION Texture;
    DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES MultipleSubresources;
    DSTORAGE_DESTINATION_TILES Tiles;
    DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES_RANGE MultipleSubresourcesRange;
};
struct DSTORAGE_REQUEST {
    DSTORAGE_REQUEST_OPTIONS Options;
    DSTORAGE_SOURCE Source;
    DSTORAGE_DESTINATION Destination;
    UINT32 UncompressedSize;
    UINT64 CancellationTag;
    const CHAR *Name;
};

#define DSTORAGE_REQUEST_MAX_NAME 64
enum DSTORAGE_COMMAND_TYPE {
    DSTORAGE_COMMAND_TYPE_NONE = -1,
    DSTORAGE_COMMAND_TYPE_REQUEST = 0,
    DSTORAGE_COMMAND_TYPE_STATUS = 1,
    DSTORAGE_COMMAND_TYPE_SIGNAL = 2,
    DSTORAGE_COMMAND_TYPE_EVENT = 3,
};
struct DSTORAGE_ERROR_PARAMETERS_REQUEST {
    WCHAR Filename[MAX_PATH];
    CHAR RequestName[DSTORAGE_REQUEST_MAX_NAME];
    DSTORAGE_REQUEST Request;
};
struct DSTORAGE_ERROR_PARAMETERS_STATUS { IDStorageStatusArray *StatusArray; UINT32 Index; };
struct DSTORAGE_ERROR_PARAMETERS_SIGNAL { ID3D12Fence *Fence; UINT64 Value; };
struct DSTORAGE_ERROR_PARAMETERS_EVENT { HANDLE Handle; };
struct DSTORAGE_ERROR_FIRST_FAILURE {
    HRESULT HResult;
    DSTORAGE_COMMAND_TYPE CommandType;
    union {
        DSTORAGE_ERROR_PARAMETERS_REQUEST Request;
        DSTORAGE_ERROR_PARAMETERS_STATUS Status;
        DSTORAGE_ERROR_PARAMETERS_SIGNAL Signal;
        DSTORAGE_ERROR_PARAMETERS_EVENT Event;
    };
};
struct DSTORAGE_ERROR_RECORD {
    UINT32 FailureCount;
    DSTORAGE_ERROR_FIRST_FAILURE FirstFailure;
};

struct IDStorageFactory : public IUnknown {
    virtual HRESULT STDMETHODCALLTYPE CreateQueue(const DSTORAGE_QUEUE_DESC *desc, REFIID riid,
                                                  void **ppv) = 0;
    virtual HRESULT STDMETHODCALLTYPE OpenFile(const WCHAR *path, REFIID riid, void **ppv) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateStatusArray(UINT32 capacity, PCSTR name, REFIID riid,
                                                        void **ppv) = 0;
    virtual void STDMETHODCALLTYPE SetDebugFlags(UINT32 flags) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetStagingBufferSize(UINT32 size) = 0;
};
struct IDStorageQueue : public IUnknown {
    virtual void STDMETHODCALLTYPE EnqueueRequest(const DSTORAGE_REQUEST *request) = 0;
    virtual void STDMETHODCALLTYPE EnqueueStatus(IDStorageStatusArray *statusArray, UINT32 index) = 0;
    virtual void STDMETHODCALLTYPE EnqueueSignal(ID3D12Fence *fence, UINT64 value) = 0;
    virtual void STDMETHODCALLTYPE Submit() = 0;
    virtual void STDMETHODCALLTYPE CancelRequestsWithTag(UINT64 mask, UINT64 value) = 0;
    virtual void STDMETHODCALLTYPE Close() = 0;
    virtual HANDLE STDMETHODCALLTYPE GetErrorEvent() = 0;
    virtual void STDMETHODCALLTYPE RetrieveErrorRecord(DSTORAGE_ERROR_RECORD *record) = 0;
    virtual void STDMETHODCALLTYPE Query(DSTORAGE_QUEUE_INFO *info) = 0;
};

// Interface IDs from dstorage.h.
static const GUID IID_IDStorageFactory_ = {
    0x6924ea0c, 0xc3cd, 0x4826, {0xb1, 0x0a, 0xf6, 0x4f, 0x4e, 0xd9, 0x27, 0xc1}};
static const GUID IID_IDStorageFile_ = {
    0x5de95e7b, 0x955a, 0x4868, {0xa7, 0x3c, 0x24, 0x3b, 0x29, 0xf4, 0xb8, 0xda}};
static const GUID IID_IDStorageQueue_ = {
    0xcfdbd83f, 0x9e06, 0x4fda, {0x8e, 0xa5, 0x69, 0x04, 0x21, 0x37, 0xf4, 0x9b}};

typedef HRESULT(WINAPI *PFN_DStorageGetFactory)(REFIID riid, void **ppv);

static_assert(sizeof(DSTORAGE_REQUEST_OPTIONS) == 16, "DirectStorage request options layout");
static_assert(sizeof(DSTORAGE_QUEUE_DESC) == 32, "DirectStorage queue descriptor layout");
