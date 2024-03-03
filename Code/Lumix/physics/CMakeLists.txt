add_library(physics)

welder_setup_library(physics ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(physics ${CMAKE_CURRENT_LIST_DIR})

set(physics_src
    ${CMAKE_CURRENT_LIST_DIR}/physics_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/physics_module.h
    ${CMAKE_CURRENT_LIST_DIR}/physics_resources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/physics_resources.h
    ${CMAKE_CURRENT_LIST_DIR}/physics_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/physics_system.h
)

if(BUILD_EDITOR)
  set(physics_src ${physics_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/physics_plugins.cpp
  )
endif()

target_sources(physics
  PRIVATE
#    ${physics_src}
    ${CMAKE_CURRENT_LIST_DIR}/PhysicsStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/PhysicsStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

target_link_libraries(physics
  PUBLIC
    Common
    Zilch
    foundation
	  engine
)
