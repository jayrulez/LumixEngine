add_library(editor)

welder_setup_library(editor ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(editor ${CMAKE_CURRENT_LIST_DIR})

set(editor_src
    ${CMAKE_CURRENT_LIST_DIR}/asset_browser.cpp
    ${CMAKE_CURRENT_LIST_DIR}/asset_browser.h
    ${CMAKE_CURRENT_LIST_DIR}/asset_compiler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/asset_compiler.h
    ${CMAKE_CURRENT_LIST_DIR}/editor_asset.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor_asset.h
    ${CMAKE_CURRENT_LIST_DIR}/entity_folders.cpp
    ${CMAKE_CURRENT_LIST_DIR}/entity_folders.h
    ${CMAKE_CURRENT_LIST_DIR}/file_system_watcher.h
    ${CMAKE_CURRENT_LIST_DIR}/gizmo.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gizmo.h
    ${CMAKE_CURRENT_LIST_DIR}/log_ui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/log_ui.h
    ${CMAKE_CURRENT_LIST_DIR}/prefab_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/prefab_system.h
    ${CMAKE_CURRENT_LIST_DIR}/profiler_ui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/profiler_ui.h
    ${CMAKE_CURRENT_LIST_DIR}/property_grid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/property_grid.h
    ${CMAKE_CURRENT_LIST_DIR}/render_interface.h
    ${CMAKE_CURRENT_LIST_DIR}/settings.cpp
    ${CMAKE_CURRENT_LIST_DIR}/settings.h
    ${CMAKE_CURRENT_LIST_DIR}/spline_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/spline_editor.h
    ${CMAKE_CURRENT_LIST_DIR}/studio_app.cpp
    ${CMAKE_CURRENT_LIST_DIR}/studio_app.h
    ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils.h
    ${CMAKE_CURRENT_LIST_DIR}/world_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/world_editor.h

)

if (${WELDER_PLATFORM} STREQUAL "Windows")
  set(editor_src ${editor_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/file_system_watcher.cpp
  )
endif()

if (${WELDER_PLATFORM} STREQUAL "Linux")
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
  ${WELDER_ROOT_DIR}/External/luau/lib/win
)

target_link_libraries(editor
  PUBLIC
    Common
    Zilch
    foundation
	  engine
    winmm
    Luau
)

target_include_directories(editor
  PUBLIC
  ${WELDER_ROOT_DIR}/External
)
