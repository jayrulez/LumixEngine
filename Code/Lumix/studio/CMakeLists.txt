add_executable(studio)

zero_setup_library(studio ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(studio ${CMAKE_CURRENT_LIST_DIR})

set(studio_src
)

if (${ZERO_PLATFORM} STREQUAL "Windows")
  set(studio_src ${studio_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/main.cpp
#    ${CMAKE_CURRENT_LIST_DIR}/win/resource.rc
  )
endif()

if (${ZERO_PLATFORM} STREQUAL "Linux")
  set(studio_src ${studio_src}
    ${CMAKE_CURRENT_LIST_DIR}/linux/main.cpp
  )
endif()

target_sources(studio
  PRIVATE
    ${studio_src}
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

target_link_libraries(studio
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
