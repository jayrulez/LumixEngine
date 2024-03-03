add_library(engine)

welder_setup_library(engine ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(engine ${CMAKE_CURRENT_LIST_DIR})

set(engine_src
    ${CMAKE_CURRENT_LIST_DIR}/controller_device.h
    ${CMAKE_CURRENT_LIST_DIR}/core.cpp
    ${CMAKE_CURRENT_LIST_DIR}/core.h
    ${CMAKE_CURRENT_LIST_DIR}/engine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/engine.def
    ${CMAKE_CURRENT_LIST_DIR}/engine.h
    ${CMAKE_CURRENT_LIST_DIR}/input_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/input_system.h
    ${CMAKE_CURRENT_LIST_DIR}/lua_api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_wrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lua_wrapper.h
    ${CMAKE_CURRENT_LIST_DIR}/lumix.h
    ${CMAKE_CURRENT_LIST_DIR}/plugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plugin.h
    ${CMAKE_CURRENT_LIST_DIR}/prefab.cpp
    ${CMAKE_CURRENT_LIST_DIR}/prefab.h
    ${CMAKE_CURRENT_LIST_DIR}/reflection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reflection.h
    ${CMAKE_CURRENT_LIST_DIR}/resource.cpp
    ${CMAKE_CURRENT_LIST_DIR}/resource.h
    ${CMAKE_CURRENT_LIST_DIR}/resource_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/resource_manager.h
    ${CMAKE_CURRENT_LIST_DIR}/world.cpp
    ${CMAKE_CURRENT_LIST_DIR}/world.h
    
)

if (${WELDER_PLATFORM} STREQUAL "Windows")
  set(engine_src ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/controller_device.cpp
  )
endif()

if (${WELDER_PLATFORM} STREQUAL "Linux")
  set(engine_src ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/linux/controller_device.cpp
  )
endif()

target_sources(engine
  PRIVATE
#    ${engine_src}
    ${CMAKE_CURRENT_LIST_DIR}/EngineStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/EngineStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

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
    foundation
)
