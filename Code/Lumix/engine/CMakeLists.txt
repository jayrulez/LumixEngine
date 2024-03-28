add_library(engine)

zero_setup_library(engine ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(engine ${CMAKE_CURRENT_LIST_DIR})

set(LUAU_LIB_DIR ${ZERO_ROOT_DIR}/External/luau/lib/win)
set(LUAU_INC_DIR ${ZERO_ROOT_DIR}/External/luau/include)

add_library(Lumix::Luau SHARED IMPORTED GLOBAL)
set_target_properties(Lumix::Luau PROPERTIES IMPORTED_LOCATION "${LUAU_LIB_DIR}/Luau.dll")
set_target_properties(Lumix::Luau PROPERTIES IMPORTED_IMPLIB "${LUAU_LIB_DIR}/Luau.lib")
set_target_properties(Lumix::Luau PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LUAU_INC_DIR}")

function(zero_link_target_luau TARGET_NAME)

    target_link_libraries(${TARGET_NAME} PRIVATE Lumix::Luau)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Lumix::Luau> $<TARGET_FILE_DIR:${TARGET_NAME}>
    )

endfunction()

set(engine_src
    ${CMAKE_CURRENT_LIST_DIR}/controller_device.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core.cpp
    ${CMAKE_CURRENT_LIST_DIR}/core.hpp
    ${CMAKE_CURRENT_LIST_DIR}/engine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/engine.def
    ${CMAKE_CURRENT_LIST_DIR}/engine.hpp
    ${CMAKE_CURRENT_LIST_DIR}/file_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/file_system.hpp
    ${CMAKE_CURRENT_LIST_DIR}/input_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/input_system.hpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_wrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_wrapper.hpp
    ${CMAKE_CURRENT_LIST_DIR}/lumix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/plugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plugin.hpp
    ${CMAKE_CURRENT_LIST_DIR}/prefab.cpp
    ${CMAKE_CURRENT_LIST_DIR}/prefab.hpp
    ${CMAKE_CURRENT_LIST_DIR}/reflection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reflection.hpp
    ${CMAKE_CURRENT_LIST_DIR}/resource.cpp
    ${CMAKE_CURRENT_LIST_DIR}/resource.hpp
    ${CMAKE_CURRENT_LIST_DIR}/resource_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/resource_manager.hpp
    ${CMAKE_CURRENT_LIST_DIR}/world.cpp
    ${CMAKE_CURRENT_LIST_DIR}/world.hpp
    
)

if (${ZERO_PLATFORM} STREQUAL "Windows")
  set(engine_src ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/controller_device.cpp
  )
endif()

if (${ZERO_PLATFORM} STREQUAL "Linux")
  set(engine_src ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/linux/controller_device.cpp
  )
endif()

set(engine_src ${engine_src}
  ${ZERO_ROOT_DIR}/External/lz4/lz4.h
  ${ZERO_ROOT_DIR}/External/lz4/lz4.c

  ${ZERO_ROOT_DIR}/External/imgui/IconsFontAwesome5.h
  ${ZERO_ROOT_DIR}/External/imgui/imconfig.h
#  ${ZERO_ROOT_DIR}/External/imgui/imgui.cpp
  ${ZERO_ROOT_DIR}/External/imgui/imgui.h
#  ${ZERO_ROOT_DIR}/External/imgui/imgui_demo.cpp
#  ${ZERO_ROOT_DIR}/External/imgui/imgui_draw.cpp
#  ${ZERO_ROOT_DIR}/External/imgui/imgui_freetype.cpp
  ${ZERO_ROOT_DIR}/External/imgui/imgui_freetype.h
  ${ZERO_ROOT_DIR}/External/imgui/imgui_internal.h
#  ${ZERO_ROOT_DIR}/External/imgui/imgui_tables.cpp
  ${ZERO_ROOT_DIR}/External/imgui/imgui_unity.cpp
  ${ZERO_ROOT_DIR}/External/imgui/imgui_user.h
  ${ZERO_ROOT_DIR}/External/imgui/imgui_user.inl
#  ${ZERO_ROOT_DIR}/External/imgui/imgui_widgets.cpp
  ${ZERO_ROOT_DIR}/External/imgui/imstb_rectpack.h
  ${ZERO_ROOT_DIR}/External/imgui/imstb_textedit.h
  ${ZERO_ROOT_DIR}/External/imgui/imstb_truetype.h
)

target_sources(engine
  PRIVATE
    ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/EngineStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/EngineStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

zero_source_ignore_precompiled_header(${ZERO_ROOT_DIR}/External/lz4/lz4.c)

set_source_files_properties(${ZERO_ROOT_DIR}/External/lz4/lz4.c PROPERTIES LANGUAGE CXX)

target_link_directories(engine
  PUBLIC
  ${ZERO_ROOT_DIR}/External/luau/lib/win
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
    core
    Lumix::Luau
)

zero_link_target_luau(engine)

target_include_directories(engine
  PUBLIC
  ${ZERO_ROOT_DIR}/External
  ${ZERO_ROOT_DIR}/External/luau/include
  ${ZERO_ROOT_DIR}/External/freetype/include
)
