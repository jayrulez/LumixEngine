add_library(engine)

welder_setup_library(engine ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(engine ${CMAKE_CURRENT_LIST_DIR})

set(engine_src
    ${CMAKE_CURRENT_LIST_DIR}/controller_device.h
    ${CMAKE_CURRENT_LIST_DIR}/core.cpp
    ${CMAKE_CURRENT_LIST_DIR}/core.h
    ${CMAKE_CURRENT_LIST_DIR}/engine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/engine.def
    ${CMAKE_CURRENT_LIST_DIR}/engine.h
    ${CMAKE_CURRENT_LIST_DIR}/input_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/input_system.h
    ${CMAKE_CURRENT_LIST_DIR}/lua_api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_wrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_wrapper.h
    ${CMAKE_CURRENT_LIST_DIR}/lumix.h
    ${CMAKE_CURRENT_LIST_DIR}/plugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plugin.h
    ${CMAKE_CURRENT_LIST_DIR}/prefab.cpp
    ${CMAKE_CURRENT_LIST_DIR}/prefab.h
    ${CMAKE_CURRENT_LIST_DIR}/reflection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reflection.h
    ${CMAKE_CURRENT_LIST_DIR}/resource.cpp
    ${CMAKE_CURRENT_LIST_DIR}/resource.h
    ${CMAKE_CURRENT_LIST_DIR}/resource_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/resource_manager.h
    ${CMAKE_CURRENT_LIST_DIR}/world.cpp
    ${CMAKE_CURRENT_LIST_DIR}/world.h
    
)

if (${WELDER_PLATFORM} STREQUAL "Windows")
  set(engine_src ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/controller_device.cpp
  )
endif()

if (${WELDER_PLATFORM} STREQUAL "Linux")
  set(engine_src ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/linux/controller_device.cpp
  )
endif()

set(engine_src ${engine_src}
  ${WELDER_ROOT_DIR}/external/lz4/lz4.h
  ${WELDER_ROOT_DIR}/external/lz4/lz4.c

  ${WELDER_ROOT_DIR}/external/imgui/IconsFontAwesome5.h
  ${WELDER_ROOT_DIR}/external/imgui/imconfig.h
#  ${WELDER_ROOT_DIR}/external/imgui/imgui.cpp
  ${WELDER_ROOT_DIR}/external/imgui/imgui.h
#  ${WELDER_ROOT_DIR}/external/imgui/imgui_demo.cpp
#  ${WELDER_ROOT_DIR}/external/imgui/imgui_draw.cpp
#  ${WELDER_ROOT_DIR}/external/imgui/imgui_freetype.cpp
  ${WELDER_ROOT_DIR}/external/imgui/imgui_freetype.h
  ${WELDER_ROOT_DIR}/external/imgui/imgui_internal.h
#  ${WELDER_ROOT_DIR}/external/imgui/imgui_tables.cpp
  ${WELDER_ROOT_DIR}/external/imgui/imgui_unity.cpp
  ${WELDER_ROOT_DIR}/external/imgui/imgui_user.h
  ${WELDER_ROOT_DIR}/external/imgui/imgui_user.inl
#  ${WELDER_ROOT_DIR}/external/imgui/imgui_widgets.cpp
  ${WELDER_ROOT_DIR}/external/imgui/imstb_rectpack.h
  ${WELDER_ROOT_DIR}/external/imgui/imstb_textedit.h
  ${WELDER_ROOT_DIR}/external/imgui/imstb_truetype.h
)

target_sources(engine
  PRIVATE
    ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/EngineStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/EngineStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

welder_source_ignore_precompiled_header(${WELDER_ROOT_DIR}/external/lz4/lz4.c)

set_source_files_properties(${WELDER_ROOT_DIR}/external/lz4/lz4.c PROPERTIES LANGUAGE CXX)

target_link_directories(engine
  PUBLIC
  ${WELDER_ROOT_DIR}/external/luau/lib/win
)

target_link_libraries(engine
  PUBLIC
    Common
    Geometry
    Meta
    Platform
    Serialization
    SpatialPartition
    Zilch
    foundation
    Luau
)

target_include_directories(engine
  PUBLIC
  ${WELDER_ROOT_DIR}/external
  ${WELDER_ROOT_DIR}/external/luau/include
  ${WELDER_ROOT_DIR}/external/freetype/include
)
