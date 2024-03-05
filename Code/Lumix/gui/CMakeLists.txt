add_library(gui)

welder_setup_library(gui ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(gui ${CMAKE_CURRENT_LIST_DIR})

set(gui_src
    ${CMAKE_CURRENT_LIST_DIR}/gui_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gui_module.h
    ${CMAKE_CURRENT_LIST_DIR}/gui_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gui_system.h
    ${CMAKE_CURRENT_LIST_DIR}/sprite.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sprite.h
)

if(BUILD_EDITOR)
  set(gui_src ${gui_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/gui_plugins.cpp
  )
endif()

target_sources(gui
  PRIVATE
    ${gui_src}
    ${CMAKE_CURRENT_LIST_DIR}/GuiStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/GuiStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

target_link_libraries(gui
  PUBLIC
    Common
    Zilch
    foundation
	  engine
    renderer
)

if(BUILD_EDITOR)
  target_link_libraries(gui
    PUBLIC
      editor
  )
endif()
