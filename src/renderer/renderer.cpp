#include "renderer.h"

#include "core/arena_allocator.h"
#include "core/array.h"
#include "core/atomic.h"
#include "core/command_line_parser.h"
#include "core/hash.h"
#include "core/log.h"
#include "core/os.h"
#include "core/page_allocator.h"
#include "core/profiler.h"
#include "core/stack_array.h"
#include "core/string.h"
#include "core/sync.h"
#include "core/thread.h"
#include "engine/engine.h"
#include "engine/reflection.h"
#include "engine/resource_manager.h"
#include "engine/world.h"
#include "renderer/draw_stream.h"
#include "renderer/font.h"
#include "renderer/material.h"
#include "renderer/model.h"
#include "renderer/particle_system.h"
#include "renderer/pipeline.h"
#include "renderer/postprocess.h"
#include "renderer/render_module.h"
#include "renderer/shader.h"
#include "renderer/terrain.h"
#include "renderer/texture.h"

namespace Lumix {

static const ComponentType MODEL_INSTANCE_TYPE = reflection::getComponentType("model_instance");

RenderBufferHandle RenderPlugin::renderBeforeTonemap(const GBuffer& gbuffer, RenderBufferHandle input, Pipeline& pipeline) {
	return input;
}
RenderBufferHandle RenderPlugin::renderBeforeTransparent(const GBuffer& gbuffer, RenderBufferHandle input, Pipeline& pipeline) {
	return input;
}
RenderBufferHandle RenderPlugin::renderAfterTonemap(const GBuffer& gbuffer, RenderBufferHandle input, Pipeline& pipeline) {
	return input;
}
bool RenderPlugin::tonemap(RenderBufferHandle input, RenderBufferHandle& output, Pipeline& pipeline) {
	return false;
}
bool RenderPlugin::debugOutput(RenderBufferHandle input, Pipeline& pipeline) {
	return false;
}
RenderBufferHandle RenderPlugin::renderAA(const GBuffer& gbuffer, RenderBufferHandle input, Pipeline& pipeline) {
	return INVALID_RENDERBUFFER;
}

void initFSR3(Renderer& renderer, IAllocator& allocator);

struct Renderbuffer {
	// The buffer is created in the ACTIVE state.
	// Once the user no longer needs the buffer, it can be marked as REUSABLE.
	// At the end of a frame, every REUSABLE buffer is marked as TO_REMOVE,
	// and every TO_REMOVE buffer is released.
	// This ensures that unused buffers are not kept around longer than necessary,
	// and buffers can be reused instead of being destroyed and recreated within the same frame.
	enum State { ACTIVE, REUSABLE, TO_REMOVE };

#ifdef LUMIX_DEBUG
	StaticString<32> debug_name;
#endif
	gpu::TextureHandle handle;
	IVec2 size;
	gpu::TextureFormat format;
	gpu::TextureFlags flags;
	State state;
};

template <u32 ALIGN> struct TransientBuffer {
	static constexpr u32 INIT_SIZE = 1024 * 1024;

	void init() {
		m_buffer = gpu::allocBufferHandle();
		m_offset = 0;
		gpu::createBuffer(m_buffer, gpu::BufferFlags::MAPPABLE, INIT_SIZE, nullptr, "transient");
		m_size = INIT_SIZE;
		m_ptr = (u8*)gpu::map(m_buffer, INIT_SIZE);
		memset(m_ptr, 0, INIT_SIZE);
	}

	Renderer::TransientSlice alloc(u32 size) {
		Renderer::TransientSlice slice;
		size = (size + (ALIGN - 1)) & ~(ALIGN - 1);
		slice.offset = m_offset;
		slice.size = size;

		if (slice.offset + size > m_size) {
			// Grow buffer if needed
			gpu::destroy(m_buffer);
			m_size = nextPow2(slice.offset + size);
			m_buffer = gpu::allocBufferHandle();
			gpu::createBuffer(m_buffer, gpu::BufferFlags::MAPPABLE, m_size, nullptr, "transient");
			m_ptr = (u8*)gpu::map(m_buffer, m_size);
			memset(m_ptr, 0, m_size);
		}

		slice.buffer = m_buffer;
		slice.ptr = m_ptr + slice.offset;
		m_offset += size;
		return slice;
	}

	void reset() { m_offset = 0; }

	gpu::BufferHandle m_buffer = gpu::INVALID_BUFFER;
	u32 m_offset = 0;
	u32 m_size = 0;
	u8* m_ptr = nullptr;
};

// Simplified frame data - no complex job system synchronization
struct FrameData {
	FrameData(IAllocator& allocator)
		: arena_allocator(1024 * 1024 * 64, allocator, "frame data")
		, draw_stream(nullptr)			   // Will be set in constructor
		, begin_frame_draw_stream(nullptr) // Will be set in constructor
		, end_frame_draw_stream(nullptr)   // Will be set in constructor
	{}

	TransientBuffer<16> transient_buffer;
	TransientBuffer<256> uniform_buffer;
	u32 gpu_frame = 0xffFFffFF;
	bool gpu_ready = true;

	ArenaAllocator arena_allocator;
	u32 frame_number = 0;
	DrawStream* draw_stream;
	DrawStream* begin_frame_draw_stream;
	DrawStream* end_frame_draw_stream;
};

template <typename T> struct RenderResourceManager : ResourceManager {
	RenderResourceManager(const char* type_name, Renderer& renderer, IAllocator& allocator)
		: ResourceManager(allocator)
		, m_renderer(renderer)
		, m_allocator(allocator, type_name) {}

	Resource* createResource(const Path& path) override { return LUMIX_NEW(m_allocator, T)(path, *this, m_renderer, m_allocator); }

	void destroyResource(Resource& resource) override { LUMIX_DELETE(m_allocator, &resource); }

	Renderer& m_renderer;
	TagAllocator m_allocator;
};

struct GPUProfiler {
	struct Query {
		StaticString<32> name;
		gpu::QueryHandle handle;
		gpu::QueryHandle stats = gpu::INVALID_QUERY;
		u64 result;
		i64 profiler_link;
		bool is_end;
		bool is_frame;
	};

	GPUProfiler(IAllocator& allocator)
		: m_queries(allocator)
		, m_pool(allocator)
		, m_stats_pool(allocator)
		, m_gpu_to_cpu_offset(0) {}

	~GPUProfiler() {
		ASSERT(m_pool.empty());
		ASSERT(m_queries.empty());
	}

	u64 toCPUTimestamp(u64 gpu_timestamp) const { return u64(gpu_timestamp * (os::Timer::getFrequency() / double(gpu::getQueryFrequency()))) + m_gpu_to_cpu_offset; }

	void init() {
		PROFILE_FUNCTION();
		gpu::QueryHandle q = gpu::createQuery(gpu::QueryType::TIMESTAMP);
		gpu::queryTimestamp(q);
		const u64 cpu_timestamp = os::Timer::getRawTimestamp();

		u32 try_num = 0;
		while (!gpu::isQueryReady(q) && try_num < 10) {
			gpu::present();
			++try_num;
		}
		if (try_num == 10) {
			logError("Failed to get GPU timestamp, timings are unreliable.");
			m_gpu_to_cpu_offset = 0;
		} else {
			const u64 gpu_timestamp = gpu::getQueryResult(q);
			m_gpu_to_cpu_offset = cpu_timestamp - u64(gpu_timestamp * (os::Timer::getFrequency() / double(gpu::getQueryFrequency())));
			gpu::destroy(q);
		}
	}

	void clear() {
		for (const Query& q : m_queries) {
			if (!q.is_frame) gpu::destroy(q.handle);
		}
		m_queries.clear();

		for (const gpu::QueryHandle h : m_pool) {
			gpu::destroy(h);
		}
		m_pool.clear();

		if (m_stats_query) gpu::destroy(m_stats_query);
		m_stats_query = gpu::INVALID_QUERY;

		for (const gpu::QueryHandle h : m_stats_pool) {
			gpu::destroy(h);
		}
		m_stats_pool.clear();
	}

	gpu::QueryHandle allocQuery() {
		if (!m_pool.empty()) {
			const gpu::QueryHandle res = m_pool.back();
			m_pool.pop();
			return res;
		}
		return gpu::createQuery(gpu::QueryType::TIMESTAMP);
	}

	gpu::QueryHandle allocStatsQuery() {
		if (!m_stats_pool.empty()) {
			const gpu::QueryHandle res = m_stats_pool.back();
			m_stats_pool.pop();
			return res;
		}
		return gpu::createQuery(gpu::QueryType::STATS);
	}

	void beginQuery(const char* name, i64 profiler_link, bool stats) {
		MutexGuard lock(m_mutex);
		Query& q = m_queries.emplace();
		q.profiler_link = profiler_link;
		q.name = name;
		q.is_end = false;
		q.is_frame = false;
		q.handle = allocQuery();
		gpu::queryTimestamp(q.handle);
		if (stats) {
			ASSERT(m_stats_counter == 0); // nested counters are not supported
			m_stats_query = allocStatsQuery();
			gpu::beginQuery(m_stats_query);
			m_stats_counter = 1;
		} else if (m_stats_counter > 0) {
			++m_stats_counter;
		}
	}

	void endQuery() {
		MutexGuard lock(m_mutex);
		Query& q = m_queries.emplace();
		q.is_end = true;
		q.is_frame = false;
		q.handle = allocQuery();
		gpu::queryTimestamp(q.handle);
		if (m_stats_counter > 0) {
			--m_stats_counter;
			if (m_stats_counter == 0) {
				gpu::endQuery(m_stats_query);
				q.stats = m_stats_query;
				m_stats_query = gpu::INVALID_QUERY;
			}
		}
	}

	void frame() {
		PROFILE_FUNCTION();
		MutexGuard lock(m_mutex);
		while (!m_queries.empty()) {
			Query q = m_queries[0];

			if (!gpu::isQueryReady(q.handle)) break;

			if (q.is_end) {
				if (q.stats && !gpu::isQueryReady(q.stats)) break;

				const u64 timestamp = toCPUTimestamp(gpu::getQueryResult(q.handle));
				if (q.stats) {
					profiler::gpuStats(gpu::getQueryResult(q.stats));
					m_stats_pool.push(q.stats);
				}
				profiler::endGPUBlock(timestamp);
			} else {
				const u64 timestamp = toCPUTimestamp(gpu::getQueryResult(q.handle));
				profiler::beginGPUBlock(q.name, timestamp, q.profiler_link);
			}
			m_pool.push(q.handle);
			m_queries.erase(0);
		}
	}

	Array<Query> m_queries;
	Array<gpu::QueryHandle> m_pool;
	Array<gpu::QueryHandle> m_stats_pool;
	Mutex m_mutex;
	i64 m_gpu_to_cpu_offset;
	u32 m_stats_counter = 0;
	gpu::QueryHandle m_stats_query = gpu::INVALID_QUERY;
};

struct RendererImpl final : Renderer {
	explicit RendererImpl(Engine& engine)
		: m_engine(engine)
		, m_allocator(engine.getAllocator(), "renderer")
		, m_texture_manager("textures", *this, m_allocator)
		, m_model_manager("models", *this, m_allocator)
		, m_particle_emitter_manager("particle emitters", *this, m_allocator)
		, m_material_manager("materials", *this, m_allocator)
		, m_shader_manager("shaders", *this, m_allocator)
		, m_font_manager(nullptr)
		, m_shader_defines(m_allocator)
		, m_profiler(m_allocator)
		, m_layers(m_allocator)
		, m_material_buffer(m_allocator)
		, m_plugins(m_allocator)
		, m_free_sort_keys(m_allocator)
		, m_sort_key_to_mesh_map(m_allocator)
		, m_semantic_defines(m_allocator)
		, m_renderbuffers(m_allocator)
		, m_atmo(*this)
		, m_cubemap_sky(*this)
		, m_tdao(*this)
		, m_sss(*this)
		, m_film_grain(*this)
		, m_dof(*this)
		, m_bloom(*this)
		, m_ssao(*this)
		, m_taa(*this)
		, m_shader_compile_queue(m_allocator) {
		RenderModule::reflect();

		LUMIX_GLOBAL_FUNC(Model::getBoneCount);
		LUMIX_GLOBAL_FUNC(Model::getBoneName);
		LUMIX_GLOBAL_FUNC(Model::getBoneParent);

		m_shader_defines.reserve(32);

		bool try_load_renderdoc = CommandLineParser::isOn("-renderdoc");
		gpu::preinit(m_allocator, try_load_renderdoc);

		// Initialize frame data with simplified structure
		m_frames[0] = LUMIX_NEW(m_allocator, FrameData)(m_allocator);
		m_frames[1] = LUMIX_NEW(m_allocator, FrameData)(m_allocator);
		for (FrameData* frame : m_frames) {
			//new (&frame) FrameData(m_allocator);
			frame->draw_stream = LUMIX_NEW(m_allocator, DrawStream)(*this);
			frame->begin_frame_draw_stream = LUMIX_NEW(m_allocator, DrawStream)(*this);
			frame->end_frame_draw_stream = LUMIX_NEW(m_allocator, DrawStream)(*this);
		}

		addPlugin(m_cubemap_sky);
		addPlugin(m_atmo);
		addPlugin(m_tdao);
		addPlugin(m_sss);
		addPlugin(m_film_grain);
		addPlugin(m_dof);
		addPlugin(m_bloom);
		addPlugin(m_ssao);
		addPlugin(m_taa);
	}

	float getLODMultiplier() const override { return m_lod_multiplier; }
	void setLODMultiplier(float value) override { m_lod_multiplier = maximum(0.f, value); }

	void serialize(OutputMemoryStream& stream) const override {}
	bool deserialize(i32 version, InputMemoryStream& stream) override { return version == 0; }

	~RendererImpl() {
		DrawStream& stream = getEndFrameDrawStream();
		for (Renderbuffer& rb : m_renderbuffers) {
			stream.destroy(rb.handle);
			rb.handle = gpu::INVALID_TEXTURE;
		}

		m_particle_emitter_manager.destroy();
		m_texture_manager.destroy();
		m_model_manager.destroy();
		m_material_manager.destroy();
		m_shader_manager.destroy();
		m_font_manager->destroy();
		LUMIX_DELETE(m_allocator, m_font_manager);

		frame();
		frame();
		frame();
		waitForRender();

		for (FrameData* frame : m_frames) {
			gpu::destroy(frame->transient_buffer.m_buffer);
			gpu::destroy(frame->uniform_buffer.m_buffer);
			LUMIX_DELETE(m_allocator, frame->draw_stream);
			LUMIX_DELETE(m_allocator, frame->begin_frame_draw_stream);
			LUMIX_DELETE(m_allocator, frame->end_frame_draw_stream);
		}
		gpu::destroy(m_material_buffer.buffer);
		gpu::destroy(m_instanced_meshes_buffer);
		m_profiler.clear();
		gpu::present();
		gpu::present();
		gpu::present();
		gpu::shutdown();
	}

	static void add(String& res, const char* a, u32 b) {
		char tmp[32];
		toCString(b, Span(tmp));
		res.append(a, tmp, "\n");
	}

	const char* getSemanticDefines(Span<const AttributeSemantic> attributes) override {
		RuntimeHash hash(attributes.begin(), sizeof(attributes[0]) * attributes.length());
		auto iter = m_semantic_defines.find(hash);
		if (!iter.isValid()) {
			String s(m_allocator);
			u32 first_empty = attributes.length();
			for (u32 i = 0; i < attributes.length(); ++i) {
				switch (attributes[i]) {
					case AttributeSemantic::COUNT: ASSERT(false); break;
					case AttributeSemantic::NONE: first_empty = minimum(first_empty, i); break;
					case AttributeSemantic::POSITION: break;

					case AttributeSemantic::NORMAL: add(s, "#define NORMAL_ATTR ", i); break;
					case AttributeSemantic::TANGENT: add(s, "#define TANGENT_ATTR ", i); break;
					case AttributeSemantic::BITANGENT: add(s, "#define BITANGENT_ATTR ", i); break;
					case AttributeSemantic::COLOR0: add(s, "#define COLOR0_ATTR ", i); break;
					case AttributeSemantic::COLOR1: add(s, "#define COLOR1_ATTR ", i); break;
					case AttributeSemantic::JOINTS: add(s, "#define INDICES_ATTR ", i); break;
					case AttributeSemantic::WEIGHTS: add(s, "#define WEIGHTS_ATTR ", i); break;
					case AttributeSemantic::TEXCOORD0: add(s, "#define UV0_ATTR ", i); break;
					case AttributeSemantic::TEXCOORD1: add(s, "#define UV1_ATTR ", i); break;
					case AttributeSemantic::AO: add(s, "#define AO_ATTR ", i); break;
				}
			}

			add(s, "#define INSTANCE0_ATTR ", first_empty + 0);
			add(s, "#define INSTANCE1_ATTR ", first_empty + 1);
			add(s, "#define INSTANCE2_ATTR ", first_empty + 2);
			add(s, "#define INSTANCE3_ATTR ", first_empty + 3);
			add(s, "#define INSTANCE4_ATTR ", first_empty + 4);
			add(s, "#define INSTANCE5_ATTR ", first_empty + 5);

			iter = m_semantic_defines.insert(hash, static_cast<String&&>(s));
		}
		return iter.value().c_str();
	}

	void initEnd() override {
		m_bloom.init();
		m_atmo.init();
		m_cubemap_sky.init();
		m_dof.init();
		m_film_grain.init();
		m_tdao.init();
		m_sss.init();
		m_ssao.init();
		m_taa.init();
	}

	void shutdownStarted() override {
		m_bloom.shutdown();
		m_atmo.shutdown();
		m_cubemap_sky.shutdown();
		m_dof.shutdown();
		m_film_grain.shutdown();
		m_tdao.shutdown();
		m_sss.shutdown();
		m_ssao.shutdown();
		m_taa.shutdown();
	}

	void initBegin() override {
		PROFILE_FUNCTION();
		gpu::InitFlags flags = gpu::InitFlags::NONE;

		char cmd_line[4096];
		os::getCommandLine(Span(cmd_line));
		CommandLineParser cmd_line_parser(cmd_line);
		while (cmd_line_parser.next()) {
			if (cmd_line_parser.currentEquals("-debug_gpu")) {
				flags = flags | gpu::InitFlags::DEBUG;
			} else if (cmd_line_parser.currentEquals("-gpu_stable_power_state")) {
				flags = flags | gpu::InitFlags::STABLE_POWER_STATE;
			}
		}

		m_instanced_meshes_buffer = gpu::allocBufferHandle();

		PROFILE_BLOCK("init_render");
		os::WindowHandle window_handle = m_engine.getMainWindow();
		if (window_handle == os::INVALID_WINDOW) {
			logError("Trying to initialize renderer without any window");
			os::messageBox("Failed to initialize renderer. More info in log.");
		}
		if (!gpu::init(window_handle, flags)) {
			os::messageBox("Failed to initialize renderer. More info in log.");
		}

		gpu::MemoryStats mem_stats;
		if (gpu::getMemoryStats(mem_stats)) {
			logInfo("Initial GPU memory stats:\n",
				"total: ",
				(mem_stats.total_available_mem / (1024.f * 1024.f)),
				"MB\n"
				"currect: ",
				(mem_stats.current_available_mem / (1024.f * 1024.f)),
				"MB\n"
				"dedicated: ",
				(mem_stats.dedicated_vidmem / (1024.f * 1024.f)),
				"MB\n");
		}

		for (FrameData* frame : m_frames) {
			frame->transient_buffer.init();
			frame->uniform_buffer.init();
		}
		gpu::createBuffer(m_instanced_meshes_buffer, gpu::BufferFlags::SHADER_BUFFER, 64 * 1024 * 1024, nullptr, "instanced_meshes");
		m_profiler.init();

		m_current_frame = m_frames[0];

		MaterialBuffer& mb = m_material_buffer;
		const u32 MAX_MATERIAL_CONSTS_COUNT = 400;
		mb.buffer = gpu::allocBufferHandle();
		mb.map.insert(RuntimeHash(), 0);
		mb.data.resize(MAX_MATERIAL_CONSTS_COUNT);
		mb.data[0].hash = RuntimeHash();
		mb.data[0].ref_count = 1;
		mb.first_free = 1;
		for (int i = 1; i < MAX_MATERIAL_CONSTS_COUNT; ++i) {
			mb.data[i].ref_count = 0;
			mb.data[i].next_free = i + 1;
		}
		mb.data.back().next_free = -1;

		DrawStream* stream = m_current_frame->draw_stream;
		stream->createBuffer(mb.buffer, gpu::BufferFlags::NONE, Material::MAX_UNIFORMS_BYTES * MAX_MATERIAL_CONSTS_COUNT, nullptr, "materials");

		float default_mat[Material::MAX_UNIFORMS_FLOATS] = {};
		stream->update(mb.buffer, &default_mat, sizeof(default_mat));

		ResourceManagerHub& manager = m_engine.getResourceManager();
		m_texture_manager.create(Texture::TYPE, manager);
		m_model_manager.create(Model::TYPE, manager);
		m_material_manager.create(Material::TYPE, manager);
		m_particle_emitter_manager.create(ParticleSystemResource::TYPE, manager);
		m_shader_manager.create(Shader::TYPE, manager);
		m_font_manager = LUMIX_NEW(m_allocator, FontManager)(*this, m_allocator);
		m_font_manager->create(FontResource::TYPE, manager);
		m_layers.emplace("default");

		initFSR3(*this, m_allocator);
	}

	MemRef copy(const void* data, u32 size) override {
		MemRef mem = allocate(size);
		memcpy(mem.data, data, size);
		return mem;
	}

	IAllocator& getAllocator() override { return m_allocator; }

	void free(const MemRef& memory) override {
		ASSERT(memory.own);
		m_allocator.deallocate(memory.data);
	}

	MemRef allocate(u32 size) override {
		MemRef ret;
		ret.size = size;
		ret.own = true;
		ret.data = m_allocator.allocate(size, 8);
		return ret;
	}

	void beginProfileBlock(const char* name, i64 link, bool stats) override {
		gpu::pushDebugGroup(name);
		m_profiler.beginQuery(name, link, stats);
	}

	void endProfileBlock() override {
		m_profiler.endQuery();
		gpu::popDebugGroup();
	}

	TransientSlice allocTransient(u32 size) override { return m_current_frame->transient_buffer.alloc(size); }

	TransientSlice allocUniform(const void* data, u32 size) override {
		const TransientSlice slice = m_current_frame->uniform_buffer.alloc(size);
		memcpy(slice.ptr, data, size);
		return slice;
	}

	TransientSlice allocUniform(u32 size) override { return m_current_frame->uniform_buffer.alloc(size); }

	gpu::BufferHandle getMaterialUniformBuffer() override { return m_material_buffer.buffer; }

	RenderBufferHandle createRenderbuffer(const RenderbufferDesc& desc) override {
		for (Renderbuffer& rb : m_renderbuffers) {
			if (!rb.handle) continue;

			if (rb.state == Renderbuffer::ACTIVE) continue;
			if (rb.size != desc.size) continue;
			if (rb.format != desc.format) continue;
			if (rb.flags != desc.flags) continue;

			rb.state = Renderbuffer::ACTIVE;
#ifdef LUMIX_DEBUG
			rb.debug_name = desc.debug_name;
#endif
			StaticString<128> name(desc.debug_name, " ", u32(&rb - m_renderbuffers.begin()));
			getDrawStream().setDebugName(rb.handle, name);
			return RenderBufferHandle(u32(&rb - m_renderbuffers.begin()));
		}

		for (Renderbuffer& rb : m_renderbuffers) {
			if (rb.handle) continue;

			rb.handle = createTexture(desc.size.x, desc.size.y, 1, desc.format, desc.flags, Renderer::MemRef(), desc.debug_name);
			rb.state = Renderbuffer::ACTIVE;
			rb.flags = desc.flags;
			rb.format = desc.format;
			rb.size = desc.size;
#ifdef LUMIX_DEBUG
			rb.debug_name = desc.debug_name;
#endif
			return RenderBufferHandle(u32(&rb - m_renderbuffers.begin()));
		}

		Renderbuffer& rb = m_renderbuffers.emplace();
		rb.handle = createTexture(desc.size.x, desc.size.y, 1, desc.format, desc.flags, Renderer::MemRef(), desc.debug_name);
		rb.state = Renderbuffer::ACTIVE;
		rb.flags = desc.flags;
		rb.format = desc.format;
		rb.size = desc.size;
#ifdef LUMIX_DEBUG
		rb.debug_name = desc.debug_name;
#endif
		return RenderBufferHandle(m_renderbuffers.size() - 1);
	}

	void releaseRenderbuffer(RenderBufferHandle idx) override {
		if (idx == INVALID_RENDERBUFFER) return;
		m_renderbuffers[idx].state = Renderbuffer::REUSABLE;
	}

	gpu::TextureHandle toTexture(RenderBufferHandle handle) override {
		if (handle >= (u32)m_renderbuffers.size()) return gpu::INVALID_TEXTURE;
		return m_renderbuffers[handle].handle;
	}

	void setRenderTargets(Span<const RenderBufferHandle> renderbuffers, RenderBufferHandle ds = INVALID_RENDERBUFFER, gpu::FramebufferFlags flags = gpu::FramebufferFlags::NONE) override {
		DrawStream& stream = getDrawStream();
		if (ds == INVALID_RENDERBUFFER && renderbuffers.length() == 0) {
			stream.setFramebuffer(nullptr, 0, gpu::INVALID_TEXTURE, gpu::FramebufferFlags::NONE);
			return;
		}

		const IVec2 viewport_size = m_renderbuffers[ds == INVALID_RENDERBUFFER ? renderbuffers[0] : ds].size;
		gpu::TextureHandle attachments[16];
		ASSERT(renderbuffers.length() <= lengthOf(attachments));
		for (u32 i = 0; i < renderbuffers.length(); ++i) {
			attachments[i] = m_renderbuffers[renderbuffers[i]].handle;
		}
		stream.setFramebuffer(attachments, renderbuffers.length(), ds != INVALID_RENDERBUFFER ? m_renderbuffers[ds].handle : gpu::INVALID_TEXTURE, flags);
		stream.viewport(0, 0, viewport_size.x, viewport_size.y);
	}

	void clearBuffers() {
		PROFILE_FUNCTION();
		for (Renderbuffer& rb : m_renderbuffers) {
			switch (rb.state) {
				case Renderbuffer::ACTIVE: break;
				case Renderbuffer::REUSABLE: rb.state = Renderbuffer::TO_REMOVE; break;
				case Renderbuffer::TO_REMOVE:
					if (rb.handle) {
						getEndFrameDrawStream().destroy(rb.handle);
						rb.handle = gpu::INVALID_TEXTURE;
					}
					break;
			}
		}

		while (!m_renderbuffers.empty()) {
			if (m_renderbuffers.last().handle) break;
			m_renderbuffers.pop();
		}
	}

	u32 createMaterialConstants(Span<const float> data) override {
		const RuntimeHash hash(data.begin(), data.length() * sizeof(float));
		auto iter = m_material_buffer.map.find(hash);
		u32 idx;
		if (iter.isValid()) {
			idx = iter.value();
		} else {
			if (m_material_buffer.first_free == -1) {
				ASSERT(false);
				++m_material_buffer.data[0].ref_count;
				return 0;
			}
			idx = m_material_buffer.first_free;
			m_material_buffer.first_free = m_material_buffer.data[m_material_buffer.first_free].next_free;
			m_material_buffer.data[idx].ref_count = 0;
			m_material_buffer.data[idx].hash = RuntimeHash(data.begin(), data.length() * sizeof(float));
			m_material_buffer.map.insert(hash, idx);

			const u32 size = u32(data.length() * sizeof(float));
			const TransientSlice slice = m_current_frame->uniform_buffer.alloc(size);
			memcpy(slice.ptr, data.begin(), size);
			m_current_frame->draw_stream->copy(m_material_buffer.buffer, slice.buffer, idx * Material::MAX_UNIFORMS_BYTES, slice.offset, size);
		}
		++m_material_buffer.data[idx].ref_count;
		return idx;
	}

	void destroyMaterialConstants(u32 idx) override {
		--m_material_buffer.data[idx].ref_count;
		if (m_material_buffer.data[idx].ref_count > 0) return;

		const RuntimeHash hash = m_material_buffer.data[idx].hash;
		m_material_buffer.data[idx].next_free = m_material_buffer.first_free;
		m_material_buffer.first_free = idx;
		m_material_buffer.map.erase(hash);
	}

	gpu::BufferHandle getInstancedMeshesBuffer() override { return m_instanced_meshes_buffer; }

	gpu::BufferHandle createBuffer(const MemRef& memory, gpu::BufferFlags flags, const char* debug_name) override {
		gpu::BufferHandle handle = gpu::allocBufferHandle();
		if (!handle) return handle;

		DrawStream& stream = getDrawStream();
		stream.createBuffer(handle, flags, memory.size, memory.data, debug_name);
		if (memory.own) stream.freeMemory(memory.data, m_allocator);
		return handle;
	}

	u8 getLayersCount() const override { return (u8)m_layers.size(); }

	const char* getLayerName(u8 layer) const override { return m_layers[layer]; }

	u8 getLayerIdx(const char* name) override {
		for (u8 i = 0; i < m_layers.size(); ++i) {
			if (m_layers[i] == name) return i;
		}
		ASSERT(m_layers.size() < 0xff);
		m_layers.emplace(name);
		return m_layers.size() - 1;
	}

	void enableBuiltinTAA(bool enable) { m_taa.m_enabled = enable; }

	const Mesh** getSortKeyToMeshMap() const override { return m_sort_key_to_mesh_map.begin(); }

	u32 allocSortKey(Mesh* mesh) override {
		if (!m_free_sort_keys.empty()) {
			const u32 key = m_free_sort_keys.back();
			m_free_sort_keys.pop();
			ASSERT(key != 0);
			if ((u32)m_sort_key_to_mesh_map.size() < key + 1) m_sort_key_to_mesh_map.resize(key + 1);
			m_sort_key_to_mesh_map[key] = mesh;
			return key;
		}
		++m_max_sort_key;
		const u32 key = m_max_sort_key;
		ASSERT(key != 0);
		if ((u32)m_sort_key_to_mesh_map.size() < key + 1) m_sort_key_to_mesh_map.resize(key + 1);
		m_sort_key_to_mesh_map[key] = mesh;
		return key;
	}

	void freeSortKey(u32 key) override {
		if (key != 0) {
			m_free_sort_keys.push(key);
		}
	}

	u32 getMaxSortKey() const override { return m_max_sort_key; }

	gpu::TextureHandle createTexture(u32 w, u32 h, u32 depth, gpu::TextureFormat format, gpu::TextureFlags flags, const MemRef& memory, const char* debug_name) override {
		gpu::TextureHandle handle = gpu::allocTextureHandle();
		if (!handle) return handle;

		DrawStream& stream = getDrawStream();
		stream.createTexture(handle, w, h, depth, format, flags, debug_name);
		if (memory.data && memory.size) {
			ASSERT(depth == 1);
			stream.update(handle, 0, 0, 0, 0, w, h, format, memory.data, memory.size);
			ASSERT(isFlagSet(flags, gpu::TextureFlags::NO_MIPS));
		}
		if (memory.own) stream.freeMemory(memory.data, m_allocator);
		return handle;
	}

	void addPlugin(RenderPlugin& plugin) override { m_plugins.push(&plugin); }

	void removePlugin(RenderPlugin& plugin) override { m_plugins.eraseItem(&plugin); }

	Span<RenderPlugin*> getPlugins() override { return m_plugins; }

	ResourceManager& getTextureManager() override { return m_texture_manager; }
	FontManager& getFontManager() override { return *m_font_manager; }

	void createModules(World& world) override {
		UniquePtr<RenderModule> module = RenderModule::createInstance(*this, m_engine, world, m_allocator);
		world.addModule(module.move());
	}

	DrawStream& getDrawStream() override { return *m_current_frame->draw_stream; }

	DrawStream& getEndFrameDrawStream() override { return *m_current_frame->end_frame_draw_stream; }

	const char* getName() const override { return "renderer"; }
	Engine& getEngine() override { return m_engine; }
	int getShaderDefinesCount() const override { return m_shader_defines.size(); }
	const char* getShaderDefine(int define_idx) const override { return m_shader_defines[define_idx]; }

	gpu::ProgramHandle queueShaderCompile(Shader& shader, const ShaderKey& key, gpu::VertexDecl decl) override {
		ASSERT(shader.isReady());

		for (const auto& i : m_shader_compile_queue) {
			if (i.content_hash == shader.m_content_hash && key == i.key) {
				return i.program;
			}
		}
		gpu::ProgramHandle program = gpu::allocProgramHandle();
		shader.compile(program, key, decl, *m_current_frame->begin_frame_draw_stream);
		m_shader_compile_queue.push({&shader, shader.m_content_hash, decl, program, key});
		return program;
	}

	u8 getShaderDefineIdx(const char* define) override {
		MutexGuard lock(m_shader_defines_mutex);
		for (int i = 0; i < m_shader_defines.size(); ++i) {
			if (m_shader_defines[i] == define) {
				return i;
			}
		}

		if (m_shader_defines.size() >= MAX_SHADER_DEFINES) {
			ASSERT(false);
			logError("Too many shader defines.");
		}

		m_shader_defines.emplace(define);
		ASSERT(m_shader_defines.size() <= 32); // m_shader_defines are reserved in renderer constructor, so getShaderDefine() is MT safe
		return u8(m_shader_defines.size() - 1);
	}

	void render() {
		PROFILE_BLOCK("render submit");

		FrameData& frame = *m_current_frame;
		profiler::pushInt("Frame", frame.frame_number);

		gpu::MemoryStats mem_stats;
		if (gpu::getMemoryStats(mem_stats)) {
			static u32 available_counter = profiler::createCounter("Available GPU memory (MB)", 0);
			static u32 buffer_counter = profiler::createCounter("Buffer memory (MB)", 0);
			static u32 texture_counter = profiler::createCounter("Texture memory (MB)", 0);
			static u32 rt_counter = profiler::createCounter("Render target memory (MB)", 0);
			auto to_MB = [](u64 B) { return float(double(B) / (1024.0 * 1024.0)); };
			profiler::pushCounter(available_counter, to_MB(mem_stats.current_available_mem));
			profiler::pushCounter(buffer_counter, to_MB(mem_stats.buffer_mem));
			profiler::pushCounter(texture_counter, to_MB(mem_stats.texture_mem));
			profiler::pushCounter(rt_counter, to_MB(mem_stats.render_target_mem));
		}

		m_profiler.beginQuery("frame", 0, false);
		frame.begin_frame_draw_stream->run();
		frame.begin_frame_draw_stream->reset();

		{
			PROFILE_BLOCK("draw stream");
			frame.draw_stream->run();
			profiler::pushInt("Drawcalls", frame.draw_stream->num_drawcalls);

			static u32 counter_time = profiler::createCounter("GPU upload (ms/frame)", 0);
			static u32 counter_size = profiler::createCounter("GPU upload (MB/frame)", 0);
			const float upload_duration = 1000.f * float(frame.draw_stream->upload_duration / double(profiler::frequency()));
			profiler::pushCounter(counter_time, upload_duration);
			profiler::pushCounter(counter_size, frame.draw_stream->upload_size / (1024.f * 1024.f));
			frame.draw_stream->reset();
		}

		frame.end_frame_draw_stream->run();
		frame.end_frame_draw_stream->reset();

		frame.arena_allocator.reset();
		m_profiler.endQuery();

		frame.gpu_frame = gpu::present();
		frame.gpu_ready = false;
		m_profiler.frame();
	}

	ArenaAllocator& getCurrentFrameAllocator() override { return m_current_frame->arena_allocator; }

	void waitForRender() override {
		// Wait for current frame to finish on GPU
		if (!m_current_frame->gpu_ready) {
			gpu::waitFrame(m_current_frame->gpu_frame);
			m_current_frame->gpu_ready = true;
		}
	}

	u32 frameNumber() const override { return m_frame_number; }

	void frame() override {
		PROFILE_FUNCTION();

		clearBuffers();

		// Process shader compilation queue
		for (const auto& i : m_shader_compile_queue) {
			i.shader->m_programs.push({i.key, i.program});
		}
		m_shader_compile_queue.clear();

		// Reset current frame
		m_current_frame->draw_stream->useProgram(gpu::INVALID_PROGRAM);
		m_current_frame->draw_stream->bindIndexBuffer(gpu::INVALID_BUFFER);
		m_current_frame->draw_stream->bindVertexBuffer(0, gpu::INVALID_BUFFER, 0, 0);
		m_current_frame->draw_stream->bindVertexBuffer(1, gpu::INVALID_BUFFER, 0, 0);
		for (u32 i = 0; i < (u32)UniformBuffer::COUNT; ++i) {
			m_current_frame->draw_stream->bindUniformBuffer(i, gpu::INVALID_BUFFER, 0, 0);
		}

		// Wait for GPU if needed before rendering current frame
		if (!m_current_frame->gpu_ready) {
			gpu::waitFrame(m_current_frame->gpu_frame);
			m_current_frame->gpu_ready = true;
		}

		// Reset transient buffers for new frame
		m_current_frame->transient_buffer.reset();
		m_current_frame->uniform_buffer.reset();

		++m_frame_number;
		profiler::pushInt("Frame", m_current_frame->frame_number);
		m_current_frame->frame_number = m_frame_number;

		for (RenderPlugin* plugin : m_plugins) {
			plugin->frame(*this);
		}

		// Simple frame flip - alternate between two frames
		if (m_current_frame == m_frames[0]) {
			m_current_frame = m_frames[1];
		} else {
			m_current_frame = m_frames[0];
		}

		// Render previous frame
		render();
	}

	Engine& m_engine;
	TagAllocator m_allocator;
	Array<StaticString<32>> m_shader_defines;
	Mutex m_shader_defines_mutex;
	Array<StaticString<32>> m_layers;
	FontManager* m_font_manager;
	RenderResourceManager<Model> m_model_manager;
	RenderResourceManager<ParticleSystemResource> m_particle_emitter_manager;
	RenderResourceManager<Shader> m_shader_manager;
	RenderResourceManager<Texture> m_texture_manager;
	RenderResourceManager<Material> m_material_manager;
	Array<u32> m_free_sort_keys;
	Array<const Mesh*> m_sort_key_to_mesh_map;
	u32 m_max_sort_key = 0;
	u32 m_frame_number = 0;
	float m_lod_multiplier = 1;
	HashMap<RuntimeHash, String> m_semantic_defines;

	Array<RenderPlugin*> m_plugins;

	// Simplified frame management - just double buffering
	FrameData* m_frames[2];
	FrameData* m_current_frame = nullptr;

	GPUProfiler m_profiler;

	struct MaterialBuffer {
		MaterialBuffer(IAllocator& alloc)
			: map(alloc)
			, data(alloc) {}

		struct Data {
			Data() {}
			u32 ref_count;
			union {
				RuntimeHash hash;
				u32 next_free;
			};
		};

		gpu::BufferHandle buffer = gpu::INVALID_BUFFER;
		Array<Data> data;
		int first_free;
		HashMap<RuntimeHash, u32> map;
	} m_material_buffer;

	Array<Renderbuffer> m_renderbuffers;
	gpu::BufferHandle m_instanced_meshes_buffer = gpu::INVALID_BUFFER;

	// Shader compilation queue - simplified without jobs
	struct ShaderToCompile {
		Shader* shader;
		StableHash content_hash;
		gpu::VertexDecl decl;
		gpu::ProgramHandle program;
		ShaderKey key;
	};
	Array<ShaderToCompile> m_shader_compile_queue;

	// built-in postprocesses
	// environment
	Atmo m_atmo;
	CubemapSky m_cubemap_sky;
	// camera
	DOF m_dof;
	FilmGrain m_film_grain;
	Bloom m_bloom;
	// global
	TDAO m_tdao;
	SSS m_sss;
	SSAO m_ssao;
	TAA m_taa;
};

LUMIX_PLUGIN_ENTRY(renderer) {
	PROFILE_FUNCTION();
	return LUMIX_NEW(engine.getAllocator(), RendererImpl)(engine);
}

} // namespace Lumix