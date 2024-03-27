add_library(physics)

zero_setup_library(physics ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(physics ${CMAKE_CURRENT_LIST_DIR})

set(physics_src
    ${CMAKE_CURRENT_LIST_DIR}/physics_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/physics_module.hpp
    ${CMAKE_CURRENT_LIST_DIR}/physics_resources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/physics_resources.hpp
    ${CMAKE_CURRENT_LIST_DIR}/physics_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/physics_system.hpp
)

if(BUILD_EDITOR)
  set(physics_src ${physics_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/physics_plugins.cpp
  )
endif()


target_link_directories(physics
  PUBLIC
  ${ZERO_ROOT_DIR}/External/physx/lib/win
)

target_sources(physics
  PRIVATE
    ${physics_src}
    ${CMAKE_CURRENT_LIST_DIR}/PhysicsStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/PhysicsStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

target_link_libraries(physics
  PUBLIC
    Common
    Zilch
    core
    engine
    renderer

    FastXml_static_64
    LowLevel_static_64
    LowLevelAABB_static_64
    LowLevelDynamics_static_64
    PhysX_64
    PhysXCommon_64
    PhysXCooking_64
    PhysXExtensions_static_64
    PhysXFoundation_64
    PhysXCharacterKinematic_static_64
    PhysXPvdSDK_static_64
    PhysXTask_static_64
    PhysXVehicle_static_64
    SceneQuery_static_64
    SimulationController_static_64
)

if(BUILD_EDITOR)
  target_link_libraries(physics
    PUBLIC
      editor
  )
endif()

target_include_directories(physics
  PUBLIC
  ${ZERO_ROOT_DIR}/External/physx/include
)

target_compile_definitions(physics
  PUBLIC
    -DPX_PHYSX_CHARACTER_STATIC_LIB
)
