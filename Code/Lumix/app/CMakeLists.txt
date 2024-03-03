add_executable(app)

welder_setup_library(app ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(app ${CMAKE_CURRENT_LIST_DIR})

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
    Zilch
    foundation
    engine
)
