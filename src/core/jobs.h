#pragma once

#include "core/thread.h"
#include "core/sync.h"
#include "core/array.h"
#include "core/atomic.h"

namespace Lumix {
namespace jobs2 {

constexpr u8 ANY_WORKER = 0xff;

struct Counter {
	Counter()
		: value(0) {}

	void increment() { value.inc(); }

	void decrement() {
		if (value.dec() == 1) { // dec() returns the value BEFORE decrement
			cv.wakeup();
		}
	}

	void wait() {
		while (i32(value) > 0) { // Use conversion operator
			MutexGuard lock(mutex);
			if (i32(value) > 0) {
				cv.sleep(mutex);
			}
		}
	}

private:
	AtomicI32 value;
	Mutex mutex;
	ConditionVariable cv;
};

struct Signal {
	Signal()
		: state(0) {} // Use atomic for state

	void turnRed() {
		state = 1; // Use assignment operator
		// Note: No wakeup needed for turnRed
	}

	void turnGreen() {
		{
			MutexGuard lock(mutex);
			state = 0; // Use assignment operator
		}
		cv.wakeup();
	}

	void wait() {
		while (i32(state) != 0) { // Use conversion operator
			MutexGuard lock(mutex);
			if (i32(state) != 0) {
				cv.sleep(mutex);
			}
		}
	}

private:
	AtomicI32 state;
	Mutex mutex;
	ConditionVariable cv;
};

struct LUMIX_CORE_API JobSystem {
	static UniquePtr<JobSystem> create(IAllocator& allocator);

	virtual ~JobSystem() {}

	virtual bool init(u8 workers_count) = 0;

	virtual void shutdown() = 0;

	virtual void run(void* data, void (*task)(void*), Counter* counter, u8 worker_index) = 0;

	virtual u8 getWorkersCount() = 0;

	template <typename F> void forEach(u32 count, u32 step, const F& f) {
		if (count == 0) return;
		if (count <= step) {
			f(0, count);
			return;
		}

		const u32 steps = (count + step - 1) / step;
		const u32 num_workers = u32(getWorkersCount());
		const u32 num_jobs = steps > num_workers ? num_workers : steps;

		Counter counter;
		struct Data {
			const F* f;
			AtomicI32 offset;
			u32 step;
			u32 count;

			Data()
				: offset(0) {} // Initialize atomic
		} data;

		data.f = &f;
		data.step = step;
		data.count = count;

		ASSERT(num_jobs > 1);

		// Launch worker jobs
		for (u32 i = 0; i < num_jobs - 1; ++i) {
			run((void*)&data,
				[](void* user_ptr) {
					Data* data = (Data*)user_ptr;
					const u32 count = data->count;
					const u32 step = data->step;
					const F* f = data->f;

					for (;;) {
						const i32 idx = data->offset.add(step); // add() returns old value
						if ((u32)idx >= count) break;
						u32 to = idx + step;
						to = to > count ? count : to;
						(*f)(idx, to);
					}
				},
				&counter,
				ANY_WORKER);
		}

		// Main thread participates
		for (;;) {
			const i32 idx = data.offset.add(step);
			if ((u32)idx >= count) break;
			u32 to = idx + step;
			to = to > count ? count : to;
			f(idx, to);
		}

		counter.wait();
	}


	template <typename F> void runOnWorkers(const F& f) {
		Counter counter;

		const u32 worker_count = getWorkersCount();

		// Launch on workers (all but one)
		for (u32 i = 0; i < worker_count - 1; ++i) {
			run((void*)&f, [](void* data) { (*(const F*)data)(); }, &counter, ANY_WORKER);
		}

		// Execute on current thread
		f();

		// Wait for completion
		counter.wait();
	}
};

} // namespace jobs2
} // namespace Lumix