/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <thread>
#include <mutex>
#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Stream
//
#if defined(__HCC__) && (__hcc_minor__ < 3)
enum queue_priority
{
    priority_high = 0,
    priority_normal = 0,
    priority_low = 0
};
#else
enum queue_priority
{
    priority_high = Kalmar::priority_high,
    priority_normal = Kalmar::priority_normal,
    priority_low = Kalmar::priority_low
};
#endif

//---
hipError_t ihipStreamCreate(TlsData *tls, hipStream_t* stream, unsigned int flags, int priority) {
    ihipCtx_t* ctx = ihipGetTlsDefaultCtx();

    hipError_t e = hipSuccess;

    if (ctx) {
        if (HIP_FORCE_NULL_STREAM) {
            *stream = 0;
        } else if( NULL == stream ){
            e = hipErrorInvalidValue;
        } else {
            hc::accelerator acc = ctx->getWriteableDevice()->_acc;

            // TODO - se try-catch loop to detect memory exception?
            //
            // Note this is an execute_any_order queue, 
            // CUDA stream behavior is that all kernels submitted will automatically
            // wait for prev to complete, this behaviour will be mainatined by 
            // hipModuleLaunchKernel. execute_any_order will help 
	    // hipExtModuleLaunchKernel , which uses a special flag

            {
                // Obtain mutex access to the device critical data, release by destructor
                LockedAccessor_CtxCrit_t ctxCrit(ctx->criticalData());

#if defined(__HCC__) && (__hcc_minor__ < 3)
                auto istream = new ihipStream_t(ctx, acc.create_view(), flags);
#else
                auto istream = new ihipStream_t(ctx, acc.create_view(Kalmar::execute_any_order, Kalmar::queuing_mode_automatic, (Kalmar::queue_priority)priority), flags);
#endif

                ctxCrit->addStream(istream);
                *stream = istream;
            }
            tprintf(DB_SYNC, "hipStreamCreate, %s\n", ToString(*stream).c_str());
        }

    } else {
        e = hipErrorInvalidDevice;
    }

    return e;
}


//---
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    HIP_INIT_API(hipStreamCreateWithFlags, stream, flags);
    if(flags == hipStreamDefault || flags == hipStreamNonBlocking)
        return ihipLogStatus(ihipStreamCreate(tls, stream, flags, priority_normal));
    else
        return ihipLogStatus(hipErrorInvalidValue);
}

//---
hipError_t hipStreamCreate(hipStream_t* stream) {
    HIP_INIT_API(hipStreamCreate, stream);

    return ihipLogStatus(ihipStreamCreate(tls, stream, hipStreamDefault, priority_normal));
}

//---
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
    HIP_INIT_API(hipStreamCreateWithPriority, stream, flags, priority);

    // clamp priority to range [priority_high:priority_low]
    priority = (priority < priority_high ? priority_high : (priority > priority_low ? priority_low : priority));
    return ihipLogStatus(ihipStreamCreate(tls, stream, flags, priority));
}

//---
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    HIP_INIT_API(hipDeviceGetStreamPriorityRange, leastPriority, greatestPriority);

    if (leastPriority != NULL) *leastPriority = priority_low;
    if (greatestPriority != NULL) *greatestPriority = priority_high;
    return ihipLogStatus(hipSuccess);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
    HIP_INIT_SPECIAL_API(hipStreamWaitEvent, TRACE_SYNC, stream, event, flags);

    hipError_t e = hipSuccess;

    if (event == nullptr) {
        e = hipErrorInvalidResourceHandle;

    } else {
        auto ecd = event->locked_copyCrit(); 
        if ((ecd._state != hipEventStatusUnitialized) && (ecd._state != hipEventStatusCreated)) {
            if (HIP_SYNC_STREAM_WAIT || (HIP_SYNC_NULL_STREAM && (stream == 0))) {
                // conservative wait on host for the specified event to complete:
                // return _stream->locked_eventWaitComplete(this, waitMode);
                //
                ecd.marker().wait((event->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked
                                                                         : hc::hcWaitModeActive);
            } else {
                stream = ihipSyncAndResolveStream(stream);
                // This will use create_blocking_marker to wait on the specified queue.
                stream->locked_streamWaitEvent(ecd);
            }
        }
    }  // else event not recorded, return immediately and don't create marker.

    return ihipLogStatus(e);
};


//---
hipError_t hipStreamQuery(hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipStreamQuery, TRACE_QUERY, stream);

    // Use default stream if 0 specified:
    if (stream == hipStreamNull) {
        ihipCtx_t* device = ihipGetTlsDefaultCtx();
        stream = device->_defaultStream;
    }

    bool isEmpty = 0;

    {
        LockedAccessor_StreamCrit_t crit(stream->_criticalData);
        isEmpty = crit->_av.get_is_empty();
    }

    hipError_t e = isEmpty ? hipSuccess : hipErrorNotReady;

    return ihipLogStatus(e);
}


//---
hipError_t hipStreamSynchronize(hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipStreamSynchronize, TRACE_SYNC, stream);

    return ihipLogStatus(ihipStreamSynchronize(tls, stream));
}


//---
/**
 * @return #hipSuccess, #hipErrorInvalidResourceHandle
 */
hipError_t hipStreamDestroy(hipStream_t stream) {
    HIP_INIT_API(hipStreamDestroy, stream);

    hipError_t e = hipSuccess;

    //--- Drain the stream:
    if (stream == NULL) {
        if (!HIP_FORCE_NULL_STREAM) {
            e = hipErrorInvalidResourceHandle;
        }
    } else {
        stream->locked_wait();

        ihipCtx_t* ctx = stream->getCtx();

        if (ctx) {
            ctx->locked_removeStream(stream);
            delete stream;
        } else {
            e = hipErrorInvalidResourceHandle;
        }
    }

    return ihipLogStatus(e);
}


//---
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
    HIP_INIT_API(hipStreamGetFlags, stream, flags);

    if (flags == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    } else if (stream == hipStreamNull) {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    } else {
        *flags = stream->_flags;
        return ihipLogStatus(hipSuccess);
    }
}


//--
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
    HIP_INIT_API(hipStreamGetPriority, stream, priority);

    if (priority == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    } else if (stream == hipStreamNull) {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    } else {
#if defined(__HCC__) && (__hcc_minor__ < 3)
        *priority = 0;
#else
        LockedAccessor_StreamCrit_t crit(stream->criticalData());
        *priority = crit->_av.get_queue_priority();
#endif
        return ihipLogStatus(hipSuccess);
    }
}

bool CallbackHandler(hsa_signal_value_t value, void* cbArgs)
{
    hipError_t e = hipSuccess;

    ihipStreamCallback_t* cb = static_cast<ihipStreamCallback_t*> (cbArgs); 

    tprintf(DB_SYNC, "ihipStreamCallbackHandler wait on stream %s\n",ToString(cb->_stream).c_str());
    GET_TLS();

    cb->_callback(cb->_stream, e, cb->_userData);
   
    hsa_signal_store_relaxed(cb->_signal,0);

    delete cb;

    return false;
}
//---
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {
    HIP_INIT_API(hipStreamAddCallback, stream, callback, userData, flags);
    hipError_t e = hipSuccess;

    
    // 1. Lock the queue
    hsa_queue_t* lockedQ = static_cast<hsa_queue_t*> (stream->criticalData()._av.acquire_locked_hsa_queue());

    // 2. Allocate a singals and create barrier
    hsa_signal_t signal;
    hsa_status_t status = hsa_signal_create(1, 0, NULL, &signal);

    hsa_signal_t depSignal;
    status = hsa_signal_create(1, 0, NULL, &depSignal);

    // 3. Register callback for the event
    ihipStreamCallback_t* cb = new ihipStreamCallback_t(stream, callback, userData);
    cb->_signal = depSignal;

    hsa_amd_signal_async_handler(signal, HSA_SIGNAL_CONDITION_EQ, 1, CallbackHandler, cb);

    // create barrier
    const uint32_t queue_mask = lockedQ->size - 1;
    uint64_t index = hsa_queue_load_write_index_scacquire(lockedQ)+1;
    uint64_t nextIndex = index + 1;

    hsa_barrier_and_packet_t* barrier = &(((hsa_barrier_and_packet_t*)(lockedQ->base_address))[index&queue_mask]);
    barrier->completion_signal = signal;
   
    // 4. Create dependent barrier 
    hsa_barrier_and_packet_t* depBarrier = &(((hsa_barrier_and_packet_t*)(lockedQ->base_address))[nextIndex & queue_mask]); 

    depBarrier->dep_signal[0] = depSignal;

    unsigned fenceBits = 0; //fenceBits |= ((HSA_FENCE_SCOPE_NONE) << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE); fenceBits |= ((HSA_FENCE_SCOPE_NONE) << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    uint16_t header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE)| fenceBits| 1 << HSA_PACKET_HEADER_BARRIER;
    
    barrier->header = header;
    depBarrier->header = header;

    // 6. Trigger the doorbell
    nextIndex = nextIndex + 1;
    hsa_queue_store_write_index_relaxed(lockedQ, nextIndex);
    hsa_signal_store_relaxed(lockedQ->doorbell_signal, index+1);

    // 7. Release queue
    stream->criticalData()._av.release_locked_hsa_queue();

    // Create a thread in detached mode to handle callback
    //ihipStreamCallback_t* cb = new ihipStreamCallback_t(stream, callback, userData);
    //std::thread(ihipStreamCallbackHandler, cb).detach();

    return ihipLogStatus(e);
}
