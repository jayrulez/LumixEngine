add_library(editor)

zero_setup_library(editor ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(editor ${CMAKE_CURRENT_LIST_DIR})

set(editor_src
    ${CMAKE_CURRENT_LIST_DIR}/asset_browser.cpp
    ${CMAKE_CURRENT_LIST_DIR}/asset_browser.hpp
    ${CMAKE_CURRENT_LIST_DIR}/asset_compiler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/asset_compiler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor_asset.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor_asset.hpp
    ${CMAKE_CURRENT_LIST_DIR}/entity_folders.cpp
    ${CMAKE_CURRENT_LIST_DIR}/entity_folders.hpp
    ${CMAKE_CURRENT_LIST_DIR}/file_system_watcher.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gizmo.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gizmo.hpp
    ${CMAKE_CURRENT_LIST_DIR}/log_ui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/log_ui.hpp
    ${CMAKE_CURRENT_LIST_DIR}/prefab_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/prefab_system.hpp
    ${CMAKE_CURRENT_LIST_DIR}/profiler_ui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/profiler_ui.hpp
    ${CMAKE_CURRENT_LIST_DIR}/property_grid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/property_grid.hpp
    ${CMAKE_CURRENT_LIST_DIR}/render_interface.hpp
    ${CMAKE_CURRENT_LIST_DIR}/settings.cpp
    ${CMAKE_CURRENT_LIST_DIR}/settings.hpp
    ${CMAKE_CURRENT_LIST_DIR}/spline_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/spline_editor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/studio_app.cpp
    ${CMAKE_CURRENT_LIST_DIR}/studio_app.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils.hpp
    ${CMAKE_CURRENT_LIST_DIR}/world_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/world_editor.hpp

)

if (${ZERO_PLATFORM} STREQUAL "Windows")
  set(editor_src ${editor_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/file_system_watcher.cpp
  )
endif()

if (${ZERO_PLATFORM} STREQUAL "Linux")
  set(editor_src ${editor_src}
    ${CMAKE_CURRENT_LIST_DIR}/linux/file_system_watcher.cpp
  )
endif()

target_sources(editor
  PRIVATE
    ${editor_src}
    ${CMAKE_CURRENT_LIST_DIR}/EditorStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/EditorStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

target_link_directories(editor
  PUBLIC
  ${ZERO_ROOT_DIR}/External/luau/lib/win
)

target_link_libraries(editor
  PUBLIC
    Common
    Zilch
    core
    engine
    winmm
    Luau
)

target_include_directories(editor
  PUBLIC
  ${ZERO_ROOT_DIR}/External
)
