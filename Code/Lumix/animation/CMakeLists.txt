add_library(animation)

zero_setup_library(animation ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(animation ${CMAKE_CURRENT_LIST_DIR})

set(animation_src
    ${CMAKE_CURRENT_LIST_DIR}/animation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/animation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/animation_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/animation_module.hpp
    ${CMAKE_CURRENT_LIST_DIR}/animation_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/controller.cpp
    ${CMAKE_CURRENT_LIST_DIR}/controller.hpp
    ${CMAKE_CURRENT_LIST_DIR}/events.hpp
    ${CMAKE_CURRENT_LIST_DIR}/nodes.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nodes.hpp
    ${CMAKE_CURRENT_LIST_DIR}/property_animation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/property_animation.hpp

)

if(BUILD_EDITOR)
  set(animation_src ${animation_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/animation_plugins.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/controller_editor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/controller_editor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/editor_nodes.cpp
    ${CMAKE_CURRENT_LIST_DIR}/editor/editor_nodes.hpp
  )
endif()

target_sources(animation
  PRIVATE
    ${animation_src}
    ${CMAKE_CURRENT_LIST_DIR}/AnimationStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/AnimationStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

target_link_libraries(animation
  PUBLIC
    Common
    Zilch
    core
    engine
    renderer
)

if(BUILD_EDITOR)
  target_link_libraries(animation
    PUBLIC
      editor
  )
endif()
