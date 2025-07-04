#include "core/array.h"
#include "core/atomic.h"
#include "core/sync.h"
#include "core/thread.h"
#include "core/jobs.h"
#include "core/string.h"

namespace Lumix {

namespace jobs2 {

struct WorkerThread : Thread {
	WorkerThread(IAllocator& allocator, u32 worker_id)
		: Thread(allocator)
		, worker_id(worker_id)
		, queue(allocator)
		, should_stop(false) {}

	int task() override {
		while (i32(should_stop) == 0) { // Use conversion operator
			Job job;
			if (tryPopJob(job)) {
				job.execute();
			} else {
				// No work available, sleep briefly
				MutexGuard guard(sleep_mutex);
				cv.sleep(sleep_mutex);
			}
		}
		return 0;
	}

	struct Job {
		void (*task)(void*) = nullptr;
		void* data = nullptr;
		Counter* counter = nullptr;

		void execute() {
			if (task) task(data);
			if (counter) counter->decrement();
		}
	};

	void pushJob(const Job& job) {
		{
			MutexGuard guard(queue_mutex);
			queue.push(job);
		}
		cv.wakeup();
	}

	bool tryPopJob(Job& job) {
		Lumix::MutexGuard guard(queue_mutex);
		if (queue.empty()) return false;
		job = queue.back();
		queue.pop();
		return true;
	}

	u32 worker_id;
	Array<Job> queue;
	Lumix::Mutex queue_mutex;
	Lumix::Mutex sleep_mutex;
	ConditionVariable cv;
	AtomicI32 should_stop;
};

struct JobSystemImpl : public JobSystem {
	JobSystemImpl(IAllocator& allocator)
		: allocator(allocator)
		, workers(allocator)
		, worker_index(0) {}

	bool init(u8 workers_count) override {
		const u32 count = workers_count > 1 ? workers_count : 1;
		workers.reserve(count);

		for (u32 i = 0; i < count; ++i) {
			UniquePtr<WorkerThread> worker = UniquePtr<WorkerThread>::create(allocator, allocator, i);
			if (worker->create(StaticString<64>("Worker #", i).data, false)) {
				worker->setAffinityMask((u64)1 << i);
				workers.push(worker.move());
			}
		}

		return !workers.empty();
	}

	void run(void* data, void (*task)(void*), Counter* counter, u8 worker_index_param) override {
		if (counter) counter->increment();

		WorkerThread::Job job;
		job.data = data;
		job.task = task;
		job.counter = counter;

		if (worker_index_param == ANY_WORKER) {
			// Round-robin distribution
			const i32 old_idx = worker_index.inc(); // inc() returns old value
			const u32 idx = old_idx % workers.size();
			workers[idx]->pushJob(job);
		} else {
			const u32 idx = worker_index_param % workers.size();
			workers[idx]->pushJob(job);
		}
	}

	void shutdown() override {
		for (auto& worker : workers) {
			worker->should_stop = 1; // Use assignment operator
			worker->cv.wakeup();
		}

		for (auto& worker : workers) {
			while (!worker->isFinished()) {
				worker->wakeup();
			}
			worker->destroy();
		}
		workers.clear();
	}

	u8 getWorkersCount() override {
		return workers.size();
	}

	IAllocator& allocator;
	Array<UniquePtr<WorkerThread>> workers;
	AtomicI32 worker_index;
};

UniquePtr<JobSystem> JobSystem::create(IAllocator& allocator) {
	return UniquePtr<JobSystemImpl>::create(allocator, allocator);
}

} // namespace jobs2
} // namespace Lumix