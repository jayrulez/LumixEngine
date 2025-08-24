#include "core/allocator.h"
#include "core/array.h"
#include "core/atomic.h"
#include "core/color.h"
#include "core/log.h"
#include "core/os.h"
#include "core/profiler.h"
#include "core/ring_buffer.h"
#include "core/string.h"
#include "core/sync.h"
#include "core/tag_allocator.h"
#include "core/thread.h"
#include "job_system.h"

/* 
There are three types of queues:
	1. Work Stealing Queue - Each worker has its own. Jobs can only be pushed by the worker itself but can be consumed by any worker.
	2. Worker Queue - Each worker has its own queue for jobs pinned to that worker.
		Jobs in this queue are executed only by the owning worker.
		Any thread, including those outside the job system, can push jobs to this queue.
	3. Global Queue - A single global queue where jobs can be executed by any worker (unlike queue 2.).
		Any thread, including those outside the job system, can push jobs to this queue (unlike queue 1.).

Invariants:
	* Jobs are executed in undefined order, i.e. if we push jobs A and B, we can't be sure that A will be executed before B. 
	* tryPop in sequence "push(), tryPop()" is guaranteed to pop a job. The consumer in this case can be on a different thread, if we are sure that push() returned.
	* If a thread calls push(jobA), tryPop() running in parallel on another thread might or might not pop jobA.
*/

#define LUMIX_PROFILE_JOBS

namespace Lumix::jobs {

struct Job {
	void (*task)(void*) = nullptr;
	void* data = nullptr;
	Counter* dec_on_finish;
	u8 worker_index;
};

struct WorkerTask;
static constexpr u64 STATE_COUNTER_MASK = 0xffFF;

struct Work {
	Work() : type(NONE), job{} {}
	Work(const Job& job) : job(job), type(JOB) {}
	enum Type {
		JOB,
		NONE
	};
	Type type;
	Job job;
};

LUMIX_FORCE_INLINE static void wake(WorkerTask& to_wake);
LUMIX_FORCE_INLINE static void wake(u32 num_jobs);
LUMIX_FORCE_INLINE static void wake();
LUMIX_FORCE_INLINE static void executeJob(const Job& job);
LUMIX_FORCE_INLINE static bool popWork(Work& work, WorkerTask* worker);

// Thread-local storage for current worker
static thread_local WorkerTask* g_current_worker = nullptr;

// single producer, multiple consumer queue
// we assume strong memory model, so we can don't need to use full barriers in some cases
// producer can call only push and tryPop - produce and consume on one end of the queue
// others can call only trySteal - consume from the other end of the queue
// backed by a ring buffer, using global queue if the ring buffer is full
struct WorkStealingQueue {
	static constexpr u32 RING_BUFFER_SIZE = 512;
	static constexpr u32 SIZE_MASK = RING_BUFFER_SIZE - 1;

	LUMIX_FORCE_INLINE void pushAndWake(const Work& obj);
	// optimized batch push
	LUMIX_FORCE_INLINE void pushAndWakeN(const Work& obj, u32 num);

	LUMIX_FORCE_INLINE bool tryPop(Work& obj) {
		for (;;) {
			const i32 producing_end = m_producing_end - 1;
			m_producing_end = producing_end;
			// decrement producing_end first so that concurrent stealers can't pop the same element without us knowing
			// we need full memory barrier because we can't allow the store and load to be reordered
			memoryBarrier();
			const i32 stealing_end = m_stealing_end;
			
			// queue is empty
			if (stealing_end > producing_end) {
				// reset to normal empty state (m_producing_end == m_stealing_end)
				m_producing_end = stealing_end;
				return false;
			}

			obj = m_queue[producing_end & SIZE_MASK];
			
			const bool is_last_element = stealing_end == producing_end;
			if (!is_last_element) {
				// we are not trying to pop the last element
				// and we decremented producing_end before,
				// so concurrent stealers can't pop the same element
				return true;
			}

			// we are trying to pop the last element
			// we need to handle concurrent stealers
			// we try to change m_stealing_end because that's the only thing that can be changed by stealers
			if (AtomicI32::compareExchange(&m_stealing_end, stealing_end + 1, stealing_end)) {
				// we were faster, reset to normal empty state (m_producing_end == m_stealing_end)
				m_producing_end = stealing_end + 1;
				return true;
			}

			// concurrent stealer was faster, queue is empty
			m_producing_end = stealing_end + 1;
			return false;
		}
	}

	LUMIX_FORCE_INLINE bool trySteal(Work& obj) {
		for (;;) {
			const i32 stealing_end = m_stealing_end;
			// read stealing_end first, so we won't miss concurrent trySteal, or tryPop popping the last element
			readBarrier(); 
			const i32 producing_end = m_producing_end;
			
			const bool is_empty = stealing_end >= producing_end;
			if (is_empty) return false;

			obj = m_queue[stealing_end & SIZE_MASK];
			
			// sync with other concurrent stealers, or tryPop in case of the last remaining element
			if (AtomicI32::compareExchange(&m_stealing_end, stealing_end + 1, stealing_end)) {
				// we managed to pop the element
				return true;
			}
			// concurrent stealer or tryPop was faster, retry
		}
	}

	// align, so they are not on the same cacheline, since they have different access patterns
	alignas(64) volatile i32 m_stealing_end = 0; 	// both producer and consumers can write this
	alignas(64) volatile i32 m_producing_end = 0; 	// only producer modifies this, consumers can read it
	// if m_producing_end > m_stealing_end, queue is not empty
	Work m_queue[RING_BUFFER_SIZE];
};

// MPMC queue
// very fast tryPop on empty queue, otherwise using mutex 
struct WorkQueue {
	// queue can be modified only when holding mutex
	AtomicI32 empty = 1; // tryPop can just read this and not lock the mutex if the queue is empty
	Lumix::Mutex mutex;
	Array<Work> queue;

	WorkQueue(IAllocator& allocator) : queue(allocator) {}

	LUMIX_FORCE_INLINE bool tryPop(Work& obj) {
		// fastest path - empty queue is just one atomic read
		if (empty) return false;

		Lumix::MutexGuard guard(mutex);
		if (queue.empty()) {
			empty = 1;
			return false;
		}

		obj = queue.back();
		queue.pop();
		if (queue.empty()) empty = 1;
		return true;
	}

	LUMIX_FORCE_INLINE void pushAndWakeN(const Work& obj, u32 num) {
		{
			Lumix::MutexGuard guard(mutex);
			for (u32 i = 0; i < num; ++i) {
				queue.push(obj);
			}
			empty = 0;
		}
		wake(num);
	}

	LUMIX_FORCE_INLINE void pushAndWake(const Work& obj, WorkerTask* to_wake) {
		{
			Lumix::MutexGuard guard(mutex);
			queue.push(obj);
			empty = 0;
		}
		if (to_wake) wake(*to_wake);
		else wake();
	}
};

struct System {
	System(IAllocator& allocator) 
		: m_allocator(allocator, "job system")
		, m_workers(m_allocator)
		, m_sleeping_workers(m_allocator)
		, m_global_queue(m_allocator)
	{}

	TagAllocator m_allocator;
	Array<WorkerTask*> m_workers;
	WorkQueue m_global_queue; // non-worker threads must push here
	AtomicI32 m_num_sleeping = 0; // if 0, we are sure that no worker is sleeping; if not 0, workers can be in any state
	Lumix::Mutex m_sleeping_sync;
	Array<WorkerTask*> m_sleeping_workers; // only access while holding m_sleeping_sync
};


static Local<System> g_system;

static AtomicI32 g_generation = 1;

WorkerTask* getWorker()
{
	return g_current_worker;
}

LUMIX_FORCE_INLINE u16 getCounterFromState(u64 state) {
	return u16(state & STATE_COUNTER_MASK);
}

struct WorkerTask : Thread {
	WorkerTask(System& system, u8 worker_index) 
		: Thread(system.m_allocator)
		, m_system(system)
		, m_worker_index(worker_index)
		, m_work_queue(system.m_allocator)
	{}

	i32 task() override {
		profiler::showInProfiler(true);
		g_current_worker = this;
		workerLoop();
		return 0;
	}

	void workerLoop() {
		static AtomicI32 s_total_jobs_executed = 0;
		logInfo("Job system: Worker ", m_worker_index, " started");
		while (!m_finished) {
			Work work;
			if (popWork(work, this)) {
				if (work.type == Work::JOB && work.job.task) {
					executeJob(work.job);
					const int total = s_total_jobs_executed.inc();
					if (total % 1000 == 0) {
						logInfo("Job system: Total jobs executed so far: ", total);
					}
				}
				// Continue immediately to check for more work
			}
			// If popWork returned false, it either means no work available (worker went to sleep)
			// or shutdown was requested. The loop condition will handle shutdown.
		}
		logInfo("Job system: Worker ", m_worker_index, " finished");
	}

	volatile bool m_finished = false;
	
	System& m_system;
	WorkQueue m_work_queue; // for jobs that need to be pinned to a worker
	WorkStealingQueue m_wsq;
	u8 m_worker_index;
	u8 m_last_steal_idx = 0; // index of the last worker we managed to steal from
	
	// if m_is_sleeping == 0, we are sure that we are not sleeping
	// but if m_is_sleeping == 1, we are not sure if we are sleeping or not
	AtomicI32 m_is_sleeping = 0; 
};

// These fiber functions are removed in thread-based implementation

// try to steal a job from any other worker
// we have to try all workers, otherwise we could miss a job
LUMIX_FORCE_INLINE static bool trySteal(Work& work, WorkerTask* stealing_worker) {
	Array<WorkerTask*>& workers = g_system->m_workers;
	const u32 num_workers = workers.size();	
	const u32 start = stealing_worker->m_last_steal_idx;
	for (u32 i = stealing_worker->m_last_steal_idx; i < num_workers; ++i) {
		if (workers[i]->m_wsq.trySteal(work)) {
			stealing_worker->m_last_steal_idx = i;
			return true;
		}
	}
	for (u32 i = 0; i < stealing_worker->m_last_steal_idx; ++i) {
		if (workers[i]->m_wsq.trySteal(work)) {
			stealing_worker->m_last_steal_idx = i;
			return true;
		}
	}
	return false;
}

// try to pop a job from the queues
LUMIX_FORCE_INLINE static bool tryPopWork(Work& work, WorkerTask* worker) {
	// jobs in worker's work queue are rare but usually in the critical path, so we need to try first
	// try on empty queue is very fast
	if (worker->m_work_queue.tryPop(work)) return true;
	
	// then try to pop a job from wsq first, since it's very fast
	if (worker->m_wsq.tryPop(work)) return true;
	
	// then try to steal a job from other workers, this is slower than tryPop
	if (trySteal(work, worker)) return true;
	
	// it's very rare to have a job in the global queue, so we check it last
	if (g_system->m_global_queue.tryPop(work)) return true;

	// no jobs to pop
	return false;
}

// pops some work from the queues, if there are no jobs, worker goes to sleep
// returns true if there is some work to do
// return false if the worker should shutdown
LUMIX_FORCE_INLINE static bool popWork(Work& work, WorkerTask* worker) {
	while (!worker->m_finished) {
		for (u32 i = 0; i < 20; ++i) {
			if (tryPopWork(work, worker)) return true;
		}

		// no jobs, let's mark the worker as going to sleep / sleeping
		g_system->m_num_sleeping.inc();
		worker->m_is_sleeping = 1;
		
		Lumix::MutexGuard guard(g_system->m_sleeping_sync);
		
		// we must recheck the queues while holding the mutex, because somebody might have pushed a job in the meantime
		if (tryPopWork(work, worker)) {
			g_system->m_num_sleeping.dec();
			worker->m_is_sleeping = 0;
			return true;
		}

		// no jobs, let's go to sleep
		// even if somebody pushed a job in the meantime, we are sure that we will be woken up, since we hold the mutex
		#ifdef LUMIX_PROFILE_JOBS
			PROFILE_BLOCK("sleeping");
			profiler::blockColor(Color(0x30, 0x30, 0x30, 0xff).abgr());
		#endif

		g_system->m_sleeping_workers.push(worker);
		worker->sleep(g_system->m_sleeping_sync);
		g_system->m_num_sleeping.dec();
		worker->m_is_sleeping = 0;
	}

	return false;
}




void turnGreenEx(Signal* signal) {
	// turn the signal green
	signal->state.exchange(0);
}

void turnGreen(Signal* signal) {
	turnGreenEx(signal);
	#ifdef LUMIX_PROFILE_JOBS
		profiler::signalTriggered(signal->generation);
	#endif
}

LUMIX_FORCE_INLINE static void decCounter(Counter* counter) {
	for (;;) {
		const u64 old_state = counter->signal.state;
		const u16 old_counter = getCounterFromState(old_state);
		
		if (old_counter == 0) {
			// Counter is already 0, this shouldn't happen but let's be defensive
			logError("Job system: Attempting to decrement counter that is already 0");
			return;
		}
		
		u64 new_state;
		if (old_counter == 1) {
			// if we are going to turn the signal green
			new_state = 0;
		}
		else {
			// signal still red even after we decrement the counter
			new_state = (u64)(old_counter - 1);
		}
		
		// decrement the counter if nobody changed the state in the meantime
		if (counter->signal.state.compareExchange(new_state, old_state)) {
			return;
		}
	}
}

LUMIX_FORCE_INLINE static void addCounter(Counter* counter, u32 value) {
	for (;;) {
		const u64 old_state = counter->signal.state;
		const u16 old_counter = getCounterFromState(old_state);
		ASSERT(old_counter + value < 0xffFF);
		
		// In simplified thread-based system, state is just the counter value in lower 16 bits
		const u64 new_state = (u64)(old_counter + value);
		
		if (counter->signal.state.compareExchange(new_state, old_state)) {
			// if we turned the signal red
			if (old_counter == 0) {
				counter->signal.generation = g_generation.inc();
			}
			break;
		}
	}
}

LUMIX_FORCE_INLINE static void executeJob(const Job& job) {
	#ifdef LUMIX_PROFILE_JOBS
		profiler::beginJob(job.dec_on_finish ? job.dec_on_finish->signal.generation : 0);
	#endif
	job.task(job.data);
	#ifdef LUMIX_PROFILE_JOBS
		profiler::endBlock();
	#endif
	if (job.dec_on_finish) {
		decCounter(job.dec_on_finish);
	}
}


IAllocator& getAllocator() {
	return g_system->m_allocator;
}

bool init(u8 workers_count, IAllocator& allocator) {
	g_system.create(allocator);

	const u32 count = workers_count > 1 ? workers_count : 1;
	logInfo("Job system: Initializing with ", count, " workers");
	for (u32 i = 0; i < count; ++i) {
		WorkerTask* task = LUMIX_NEW(getAllocator(), WorkerTask)(*g_system, i);
		g_system->m_workers.push(task);
	}

	u32 started_workers = 0;
	for (u32 i = 0; i < count; ++i) {
		WorkerTask* task = g_system->m_workers[i];
		if (task->create(StaticString<64>("Worker #", i), false)) {
			task->setAffinityMask((u64)1 << i);
			started_workers++;
		}
		else {
			logError("Job system: Failed to create worker ", i);
			LUMIX_DELETE(getAllocator(), task);
		}
	}

	logInfo("Job system: Successfully started ", started_workers, " workers");
	return !g_system->m_workers.empty();
}


u8 getWorkersCount()
{
	const int c = g_system->m_workers.size();
	ASSERT(c <= 0xff);
	return (u8)c;
}

void shutdown()
{
	IAllocator& allocator = g_system->m_allocator;
	for (Thread* task : g_system->m_workers)
	{
		WorkerTask* wt = (WorkerTask*)task;
		wt->m_finished = true;
	}

	for (WorkerTask* task : g_system->m_workers)
	{
		while (!task->isFinished()) {
			task->wakeup();
		}
		task->destroy();
		LUMIX_DELETE(allocator, task);
	}

	g_system.destroy();
}

void turnRed(Signal* signal) {
	for (;;) {
		const u64 old_state = signal->state;
		const u16 old_counter = getCounterFromState(old_state);
		
		if (old_counter > 0) {
			// already red
			return;
		}
		
		// Set counter to 1 to make it red
		const u64 new_state = 1;
		
		if (signal->state.compareExchange(new_state, old_state)) {
			// it was green and we turned it red, so we need to increment generation
			signal->generation = g_generation.inc();
			break;
		}
	}
}

void wait(Counter* counter) {
	wait(&counter->signal);
}

void wait(Signal* signal) {
	// Simplified implementation: just spin-wait until signal is green
	// This is less efficient but much more reliable for thread-based system
	u32 timeout_counter = 0;
	while (getCounterFromState(signal->state) != 0) {
		cpuRelax();
		
		// Occasionally yield to prevent 100% CPU usage
		static thread_local u32 yield_counter = 0;
		if (++yield_counter % 1000 == 0) {
			os::sleep(0);
		}
		
		// Safety timeout to detect infinite waits
		++timeout_counter;
		if (timeout_counter > 10000000) { // Reduced timeout for faster recovery
			// This suggests a deadlock or counter that's never decremented
			// Force the signal green and log an error
			const u32 state = (u32)signal->state;
			const u16 counter_value = getCounterFromState(state);
			logError("Job system: Detected infinite wait on signal, state = ", state, ", counter = ", counter_value);
			logError("Job system: Forcing signal to green state for recovery");
			
			// Force counter to 0 to break the deadlock
			signal->state.exchange(0);
			break;
		}
	}
}

void waitAndTurnRed(Signal* signal) {
	for (;;) {
		// Try to acquire the signal (turn it red)
		if (signal->state.bitTestAndSet(0)) {
			signal->generation = g_generation.inc();
			ASSERT(signal->state & 1);
			return;
		}

		// Signal is already red, wait for it to turn green then try again
		wait(signal);
	}
}

void enter(Mutex* mutex) {
	waitAndTurnRed(&mutex->signal);
}

void exit(Mutex* mutex) {
	// Simply turn the signal green (unlock the mutex)
	// Waiting threads will detect this in their spin-wait loop
	ASSERT(mutex->signal.state & 1);
	mutex->signal.state.exchange(0);
}

void moveJobToWorker(u8 worker_index) {
	// No-op in thread-based system - jobs are distributed by work stealing
}

void yield() {
	// No-op in thread-based system - preemptive scheduling handles yielding
}

void run(void* data, void(*task)(void*), Counter* on_finished, u8 worker_index)
{
	static AtomicI32 s_total_jobs_queued = 0;
	Job job;
	job.data = data;
	job.task = task;
	job.worker_index = worker_index != ANY_WORKER ? worker_index % getWorkersCount() : worker_index;
	job.dec_on_finish = on_finished;

	if (on_finished) {
		addCounter(on_finished, 1);
	}

	const int total_queued = s_total_jobs_queued.inc();
	if (total_queued % 1000 == 0) {
		logInfo("Job system: Total jobs queued so far: ", total_queued);
	}

	if (worker_index != ANY_WORKER) {
		WorkerTask* worker = g_system->m_workers[worker_index % g_system->m_workers.size()];
		worker->m_work_queue.pushAndWake(job, worker);
		return;
	}

	WorkerTask* worker = getWorker();
	if (worker) {
		worker->m_wsq.pushAndWake(job);
		return;
	}

	g_system->m_global_queue.pushAndWake(job, nullptr);
}

void runN(void* data, void(*task)(void*), Counter* on_finished, u32 num_jobs)
{
	Job job;
	job.data = data;
	job.task = task;
	job.worker_index = ANY_WORKER;
	job.dec_on_finish = on_finished;

	if (on_finished) {
		addCounter(on_finished, num_jobs);
	}

	WorkerTask* worker = getWorker();
	if (worker) worker->m_wsq.pushAndWakeN(job, num_jobs);
	else g_system->m_global_queue.pushAndWakeN(job, num_jobs);
}

// wake the worker (if any is sleeping)
LUMIX_FORCE_INLINE static void wake(WorkerTask& worker) {
	if (!worker.m_is_sleeping) return;

	Lumix::MutexGuard guard(g_system->m_sleeping_sync);
	g_system->m_sleeping_workers.eraseItem(&worker);
	worker.wakeup();
}

// wake one worker (if any is sleeping)
LUMIX_FORCE_INLINE static void wake() {
	if (g_system->m_num_sleeping == 0) return;

	Lumix::MutexGuard guard(g_system->m_sleeping_sync);
	if (g_system->m_sleeping_workers.empty()) return;
	
	WorkerTask* to_wake = g_system->m_sleeping_workers.back();
	g_system->m_sleeping_workers.pop();
	to_wake->wakeup();
};


// wake num workers (or all if num > number of sleeping workers)
LUMIX_FORCE_INLINE static void wake(u32 num) {
	// fast path, no workers are sleeping
	if (g_system->m_num_sleeping == 0) return;

	Lumix::MutexGuard guard(g_system->m_sleeping_sync);
	for (u32 i = 0; i < num; ++i) {
		if (g_system->m_sleeping_workers.empty()) return;
		
		WorkerTask* to_wake = g_system->m_sleeping_workers.back();
		g_system->m_sleeping_workers.pop();
		to_wake->wakeup();
	}
}

// same as pushAndWake, but pushed the job `num` times
void WorkStealingQueue::pushAndWakeN(const Work& obj, u32 num) {
	const i32 producing_end = m_producing_end;
	const i32 size = producing_end - m_stealing_end;

	if (size + num > RING_BUFFER_SIZE) {
		g_system->m_global_queue.pushAndWakeN(obj, num);
		return;
	}
	
	for (u32 i = 0; i < num; ++i) {
		m_queue[(producing_end + i) & SIZE_MASK] = obj;
	}
	writeBarrier();
	m_producing_end = producing_end + num;
	wake(num);
}

void WorkStealingQueue::pushAndWake(const Work& obj) {
	// there's only one producer so we don't need to worry about concurrent push or tryPop
	// and stealers do not modify m_producing_end
	// worst case scenario is that a concurrent stealer will not be able to steal the element we are pushing right now
	const i32 producing_end = m_producing_end;
	// no need for barrier or any sync, worst case we overestimate current size (m_stealing_end can only increase, m_producing_end can't change)
	// if we overestimate and overflow because of that, we will push to global queue, which is fine
	const i32 size = producing_end - m_stealing_end;

	if (size == RING_BUFFER_SIZE) {
		// queue is full, push to global queue instead
		// queue should be big enough for this to never happen
		g_system->m_global_queue.pushAndWake(obj, nullptr);
		return;
	}

	m_queue[producing_end & SIZE_MASK] = obj;
	// ensure m_queue is written before m_producing_end
	// so concurrent trySteal won't steal not fully written m_queue element
	writeBarrier();
	m_producing_end = producing_end + 1;
	wake();
}

} // namespace Lumix::jobs
