add_executable(app)

zero_setup_library(app ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(app ${CMAKE_CURRENT_LIST_DIR})

set(app_src
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)

target_sources(app
  PRIVATE
  ${app_src}    
  ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp
)

target_link_libraries(app
  PUBLIC
    Common
    Geometry
    Meta
    Platform
    Serialization
    SpatialPartition
    Zilch
    Support
    core
    engine
    animation
    audio
    gui
    lua_script
    navigation
    physics
    renderer
    GL
    freetype
)

zero_copy_from_linked_libraries(app)
