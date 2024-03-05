add_library(lua_script)

welder_setup_library(lua_script ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(lua_script ${CMAKE_CURRENT_LIST_DIR})

set(lua_script_src
    ${CMAKE_CURRENT_LIST_DIR}/lua_script.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_script.h
    ${CMAKE_CURRENT_LIST_DIR}/lua_script_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_script_system.h
)

if(BUILD_EDITOR)
  set(lua_script_src ${lua_script_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/lua_script_plugins.cpp
  )
endif()

target_sources(lua_script
  PRIVATE
    ${lua_script_src}
    ${CMAKE_CURRENT_LIST_DIR}/LuaScriptStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/LuaScriptStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp
)

target_link_directories(lua_script
  PUBLIC
  ${WELDER_ROOT_DIR}/external/luau/lib/win
)

target_link_libraries(lua_script
  PUBLIC
    Common
    Zilch
    foundation
    engine
    Luau
)

target_include_directories(lua_script
  PUBLIC
  ${WELDER_ROOT_DIR}/external/luau/include
)

if(BUILD_EDITOR)
  target_link_libraries(lua_script
    PUBLIC
      editor
  )
endif()
