add_library(renderer)

welder_setup_library(renderer ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(renderer ${CMAKE_CURRENT_LIST_DIR})

set(renderer_src
    ${CMAKE_CURRENT_LIST_DIR}/culling_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/culling_system.h
    ${CMAKE_CURRENT_LIST_DIR}/draw2d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/draw2d.h
    ${CMAKE_CURRENT_LIST_DIR}/draw_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/draw_stream.h
    ${CMAKE_CURRENT_LIST_DIR}/font.cpp
    ${CMAKE_CURRENT_LIST_DIR}/font.h
    ${CMAKE_CURRENT_LIST_DIR}/material.cpp
    ${CMAKE_CURRENT_LIST_DIR}/material.h
    ${CMAKE_CURRENT_LIST_DIR}/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model.h
    ${CMAKE_CURRENT_LIST_DIR}/particle_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/particle_system.h
    ${CMAKE_CURRENT_LIST_DIR}/pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pipeline.h
    ${CMAKE_CURRENT_LIST_DIR}/pose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pose.h
    ${CMAKE_CURRENT_LIST_DIR}/render_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/render_module.h
    ${CMAKE_CURRENT_LIST_DIR}/renderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/renderer.h
    ${CMAKE_CURRENT_LIST_DIR}/shader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shader.h
    ${CMAKE_CURRENT_LIST_DIR}/terrain.cpp
    ${CMAKE_CURRENT_LIST_DIR}/terrain.h
    ${CMAKE_CURRENT_LIST_DIR}/texture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/texture.h
    ${CMAKE_CURRENT_LIST_DIR}/voxels.cpp
    ${CMAKE_CURRENT_LIST_DIR}/voxels.h

    ${CMAKE_CURRENT_LIST_DIR}/gpu/gpu.h
    ${CMAKE_CURRENT_LIST_DIR}/gpu/gpu.cpp

    ${CMAKE_CURRENT_LIST_DIR}/gpu/gl_ext.h
    ${CMAKE_CURRENT_LIST_DIR}/gpu/glcorearb.h
    ${CMAKE_CURRENT_LIST_DIR}/gpu/khrplatform.h
    ${CMAKE_CURRENT_LIST_DIR}/gpu/renderdoc_app.h

    ${CMAKE_CURRENT_LIST_DIR}/gpu/gpu_gl.cpp
)

set(renderer_src ${renderer_src}
  ${CMAKE_CURRENT_LIST_DIR}/gpu/gpu.h
  ${CMAKE_CURRENT_LIST_DIR}/gpu/gpu.cpp

  ${CMAKE_CURRENT_LIST_DIR}/gpu/gl_ext.h
  ${CMAKE_CURRENT_LIST_DIR}/gpu/glcorearb.h
  ${CMAKE_CURRENT_LIST_DIR}/gpu/khrplatform.h
  ${CMAKE_CURRENT_LIST_DIR}/gpu/renderdoc_app.h

  ${CMAKE_CURRENT_LIST_DIR}/gpu/gpu_gl.cpp  
)

if(BUILD_EDITOR)
  set(renderer_src ${renderer_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/composite_texture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/composite_texture.h
    ${CMAKE_CURRENT_LIST_DIR}/editor/editor_icon.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/editor_icon.h
    ${CMAKE_CURRENT_LIST_DIR}/editor/fbx_importer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/fbx_importer.h
    ${CMAKE_CURRENT_LIST_DIR}/editor/game_view.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/game_view.h
    ${CMAKE_CURRENT_LIST_DIR}/editor/model_meta.h
    ${CMAKE_CURRENT_LIST_DIR}/editor/particle_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/particle_editor.h
    ${CMAKE_CURRENT_LIST_DIR}/editor/render_plugins.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/scene_view.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/scene_view.h
    ${CMAKE_CURRENT_LIST_DIR}/editor/stb.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/terrain_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/terrain_editor.h
#    ${CMAKE_CURRENT_LIST_DIR}/editor/voxelizer_ui.cpp
#    ${CMAKE_CURRENT_LIST_DIR}/editor/voxelizer_ui.h
    ${CMAKE_CURRENT_LIST_DIR}/editor/world_viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/world_viewer.h


    ${WELDER_ROOT_DIR}/external/meshoptimizer/allocator.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/clusterizer.cpp
    ${WELDER_ROOT_DIR}/external/meshoptimizer/indexcodec.cpp
    ${WELDER_ROOT_DIR}/external/meshoptimizer/indexgenerator.cpp
    ${WELDER_ROOT_DIR}/external/meshoptimizer/meshoptimizer.h
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/overdrawanalyzer.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/overdrawoptimizer.cpp
    ${WELDER_ROOT_DIR}/external/meshoptimizer/simplifier.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/spatialorder.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/stripifier.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/vcacheanalyzer.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/vcacheoptimizer.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/vertexcodec.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/vertexfilter.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/vfetchanalyzer.cpp
#    ${WELDER_ROOT_DIR}/external/meshoptimizer/vfetchoptimizer.cpp

    ${WELDER_ROOT_DIR}/external/mikktspace/mikktspace.h
    ${WELDER_ROOT_DIR}/external/mikktspace/mikktspace.c
    
    ${WELDER_ROOT_DIR}/external/openfbx/libdeflate.h
    ${WELDER_ROOT_DIR}/external/openfbx/libdeflate.c
    ${WELDER_ROOT_DIR}/external/openfbx/ofbx.h
    ${WELDER_ROOT_DIR}/external/openfbx/ofbx.cpp

  )

  welder_source_ignore_precompiled_header(${WELDER_ROOT_DIR}/external/mikktspace/mikktspace.c)
  set_source_files_properties(${WELDER_ROOT_DIR}/external/mikktspace/mikktspace.c PROPERTIES LANGUAGE CXX)
  

  welder_source_ignore_precompiled_header(${WELDER_ROOT_DIR}/external/openfbx/libdeflate.c)
  set_source_files_properties(${WELDER_ROOT_DIR}/external/openfbx/libdeflate.c PROPERTIES LANGUAGE CXX)
endif()

target_sources(renderer
  PRIVATE
    ${renderer_src}
    ${CMAKE_CURRENT_LIST_DIR}/RendererStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RendererStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

target_link_directories(renderer
  PUBLIC
  ${WELDER_ROOT_DIR}/external/freetype/lib/win64_vs2017/release
)

target_link_libraries(renderer
  PUBLIC
    Common
    Zilch
    freetype
    foundation
	  engine
    lua_script
)

if(BUILD_EDITOR)
  target_link_libraries(renderer
    PUBLIC
      editor
  )
endif()

target_include_directories(renderer
  PUBLIC
  ${WELDER_ROOT_DIR}/external
  ${WELDER_ROOT_DIR}/external/freetype/include
)
