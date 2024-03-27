add_library(renderer)

zero_setup_library(renderer ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(renderer ${CMAKE_CURRENT_LIST_DIR})

set(renderer_src
    ${CMAKE_CURRENT_LIST_DIR}/culling_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/culling_system.hpp
    ${CMAKE_CURRENT_LIST_DIR}/draw2d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/draw2d.hpp
    ${CMAKE_CURRENT_LIST_DIR}/draw_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/draw_stream.hpp
    ${CMAKE_CURRENT_LIST_DIR}/font.cpp
    ${CMAKE_CURRENT_LIST_DIR}/font.hpp
    ${CMAKE_CURRENT_LIST_DIR}/material.cpp
    ${CMAKE_CURRENT_LIST_DIR}/material.hpp
    ${CMAKE_CURRENT_LIST_DIR}/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model.hpp
    ${CMAKE_CURRENT_LIST_DIR}/particle_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/particle_system.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pipeline.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pose.hpp
    ${CMAKE_CURRENT_LIST_DIR}/render_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/render_module.hpp
    ${CMAKE_CURRENT_LIST_DIR}/renderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/renderer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/shader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/terrain.cpp
    ${CMAKE_CURRENT_LIST_DIR}/terrain.hpp
    ${CMAKE_CURRENT_LIST_DIR}/texture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/texture.hpp
    ${CMAKE_CURRENT_LIST_DIR}/voxels.cpp
    ${CMAKE_CURRENT_LIST_DIR}/voxels.hpp
)

set(renderer_src ${renderer_src}
  ${CMAKE_CURRENT_LIST_DIR}/gpu/gpu.hpp
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
    ${CMAKE_CURRENT_LIST_DIR}/editor/composite_texture.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/editor_icon.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/editor_icon.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/fbx_importer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/fbx_importer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/game_view.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/game_view.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/model_meta.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/particle_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/particle_editor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/render_plugins.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/scene_view.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/scene_view.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/stb.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/terrain_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/terrain_editor.hpp
#    ${CMAKE_CURRENT_LIST_DIR}/editor/voxelizer_ui.cpp
#    ${CMAKE_CURRENT_LIST_DIR}/editor/voxelizer_ui.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/world_viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/world_viewer.hpp


    ${ZERO_ROOT_DIR}/External/meshoptimizer/allocator.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/clusterizer.cpp
    ${ZERO_ROOT_DIR}/External/meshoptimizer/indexcodec.cpp
    ${ZERO_ROOT_DIR}/External/meshoptimizer/indexgenerator.cpp
    ${ZERO_ROOT_DIR}/External/meshoptimizer/meshoptimizer.h
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/overdrawanalyzer.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/overdrawoptimizer.cpp
    ${ZERO_ROOT_DIR}/External/meshoptimizer/simplifier.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/spatialorder.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/stripifier.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/vcacheanalyzer.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/vcacheoptimizer.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/vertexcodec.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/vertexfilter.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/vfetchanalyzer.cpp
#    ${ZERO_ROOT_DIR}/External/meshoptimizer/vfetchoptimizer.cpp

    ${ZERO_ROOT_DIR}/External/mikktspace/mikktspace.h
    ${ZERO_ROOT_DIR}/External/mikktspace/mikktspace.c
    
    ${ZERO_ROOT_DIR}/External/openfbx/libdeflate.h
    ${ZERO_ROOT_DIR}/External/openfbx/libdeflate.c
    ${ZERO_ROOT_DIR}/External/openfbx/ofbx.h
    ${ZERO_ROOT_DIR}/External/openfbx/ofbx.cpp

  )

  zero_source_ignore_precompiled_header(${ZERO_ROOT_DIR}/External/mikktspace/mikktspace.c)
  set_source_files_properties(${ZERO_ROOT_DIR}/External/mikktspace/mikktspace.c PROPERTIES LANGUAGE CXX)
  

  zero_source_ignore_precompiled_header(${ZERO_ROOT_DIR}/External/openfbx/libdeflate.c)
  set_source_files_properties(${ZERO_ROOT_DIR}/External/openfbx/libdeflate.c PROPERTIES LANGUAGE CXX)
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
  ${ZERO_ROOT_DIR}/External/freetype/lib/win64_vs2017/release
)

target_link_libraries(renderer
  PUBLIC
    Common
    Zilch
    freetype
    core
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
  ${ZERO_ROOT_DIR}/External
  ${ZERO_ROOT_DIR}/External/freetype/include
)
