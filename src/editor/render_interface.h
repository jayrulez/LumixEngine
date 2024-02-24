#pragma once

#include "engine/lumix.h"

#include "renderer/gpu/gpu.h"
#include "editor/world_editor.h"

using ImTextureID = void*;
struct ImDrawData;

namespace Lumix {

struct RenderInterface {
	virtual ~RenderInterface() {}

	virtual struct AABB getEntityAABB(World& world, EntityRef entity, const DVec3& base) = 0;
	virtual ImTextureID createTexture(const char* name, const void* pixels, int w, int h) = 0;
	virtual void destroyTexture(ImTextureID handle) = 0;
	virtual ImTextureID loadTexture(const struct Path& path) = 0;
	virtual bool isValid(ImTextureID texture) = 0;
	virtual void unloadTexture(ImTextureID handle) = 0;
	virtual WorldView::RayHit castRay(World& world, const struct Ray& ray, EntityPtr ignored) = 0;
	virtual Path getModelInstancePath(World& world, EntityRef entity) = 0;
	virtual bool saveTexture(Engine& engine, const char* path_cstr, const void* pixels, int w, int h, bool upper_left_origin) = 0;
};

}