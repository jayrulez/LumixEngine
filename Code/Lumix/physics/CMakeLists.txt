add_library(physics)

zero_setup_library(physics ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(physics ${CMAKE_CURRENT_LIST_DIR})

set(PHYSX_DLL_DIR ${ZERO_ROOT_DIR}/External/physx/dll/vs2017/win64/release)
set(PHYSX_LIB_DIR ${ZERO_ROOT_DIR}/External/physx/lib/win)
set(PHYSX_INC_DIR ${ZERO_ROOT_DIR}/External/physx/include)

add_library(Lumix::PhysXFoundation_64 SHARED IMPORTED GLOBAL)
set_target_properties(Lumix::PhysXFoundation_64 PROPERTIES IMPORTED_LOCATION "${PHYSX_DLL_DIR}/PhysXFoundation_64.dll")
set_target_properties(Lumix::PhysXFoundation_64 PROPERTIES IMPORTED_IMPLIB "${PHYSX_LIB_DIR}/PhysXFoundation_64.lib")
set_target_properties(Lumix::PhysXFoundation_64 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PHYSX_INC_DIR}")

add_library(Lumix::PhysXCommon_64 SHARED IMPORTED GLOBAL)
set_target_properties(Lumix::PhysXCommon_64 PROPERTIES IMPORTED_LOCATION "${PHYSX_DLL_DIR}/PhysXCommon_64.dll")
set_target_properties(Lumix::PhysXCommon_64 PROPERTIES IMPORTED_IMPLIB "${PHYSX_LIB_DIR}/PhysXCommon_64.lib")
set_target_properties(Lumix::PhysXCommon_64 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PHYSX_INC_DIR}")
target_link_libraries(Lumix::PhysXCommon_64 INTERFACE Lumix::PhysXFoundation_64)

add_library(Lumix::PhysX_64 SHARED IMPORTED GLOBAL)
set_target_properties(Lumix::PhysX_64 PROPERTIES IMPORTED_LOCATION "${PHYSX_DLL_DIR}/PhysX_64.dll")
set_target_properties(Lumix::PhysX_64 PROPERTIES IMPORTED_IMPLIB "${PHYSX_LIB_DIR}/PhysX_64.lib")
set_target_properties(Lumix::PhysX_64 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PHYSX_INC_DIR}")
target_link_libraries(Lumix::PhysX_64 INTERFACE Lumix::PhysXCommon_64)

add_library(Lumix::PhysXCooking_64 SHARED IMPORTED GLOBAL)
set_target_properties(Lumix::PhysXCooking_64 PROPERTIES IMPORTED_LOCATION "${PHYSX_DLL_DIR}/PhysXCooking_64.dll")
set_target_properties(Lumix::PhysXCooking_64 PROPERTIES IMPORTED_IMPLIB "${PHYSX_LIB_DIR}/PhysXCooking_64.lib")
set_target_properties(Lumix::PhysXCooking_64 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PHYSX_INC_DIR}")
target_link_libraries(Lumix::PhysXCooking_64 INTERFACE Lumix::PhysXFoundation_64)

function(zero_link_target_physx TARGET_NAME)

    target_link_libraries(${TARGET_NAME} PRIVATE Lumix::PhysX_64)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Lumix::PhysXFoundation_64> $<TARGET_FILE_DIR:${TARGET_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Lumix::PhysXCommon_64> $<TARGET_FILE_DIR:${TARGET_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Lumix::PhysX_64> $<TARGET_FILE_DIR:${TARGET_NAME}>
    )

endfunction()

function(zero_link_target_physx_cooking TARGET_NAME)

    target_link_libraries(${TARGET_NAME} PRIVATE Lumix::PhysXCooking_64)

    target_sources(${TARGET_NAME} PUBLIC $<TARGET_FILE:Lumix::PhysXFoundation_64>)
    target_sources(${TARGET_NAME} PUBLIC $<TARGET_FILE:Lumix::PhysXCooking_64>)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Lumix::PhysXFoundation_64> $<TARGET_FILE_DIR:${TARGET_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Lumix::PhysXCooking_64> $<TARGET_FILE_DIR:${TARGET_NAME}>
    )

endfunction()

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
    Lumix::PhysX_64
    Lumix::PhysXCommon_64
    Lumix::PhysXCooking_64
    PhysXExtensions_static_64
    Lumix::PhysXFoundation_64
    PhysXCharacterKinematic_static_64
    PhysXPvdSDK_static_64
    PhysXTask_static_64
    PhysXVehicle_static_64
    SceneQuery_static_64
    SimulationController_static_64
)

zero_link_target_physx(physics)

if(BUILD_EDITOR)
  target_link_libraries(physics
    PUBLIC
      editor
  )

  zero_link_target_physx_cooking(physics)
endif()

target_include_directories(physics
  PUBLIC
  ${ZERO_ROOT_DIR}/External/physx/include
)

target_compile_definitions(physics
  PUBLIC
    -DPX_PHYSX_CHARACTER_STATIC_LIB
)
