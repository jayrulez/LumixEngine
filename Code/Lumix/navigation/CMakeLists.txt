add_library(navigation)

zero_setup_library(navigation ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(navigation ${CMAKE_CURRENT_LIST_DIR})

set(navigation_src
    ${CMAKE_CURRENT_LIST_DIR}/navigation_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/navigation_module.hpp
    ${CMAKE_CURRENT_LIST_DIR}/navigation_system.cpp
)

if(BUILD_EDITOR)
  set(navigation_src ${navigation_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/navigation_plugins.cpp
  )
endif()

set(navigation_src ${navigation_src}
${ZERO_ROOT_DIR}/External/recast/include/DetourAlloc.h
${ZERO_ROOT_DIR}/External/recast/include/DetourAssert.h
${ZERO_ROOT_DIR}/External/recast/include/DetourCommon.h
${ZERO_ROOT_DIR}/External/recast/include/DetourCrowd.h
${ZERO_ROOT_DIR}/External/recast/include/DetourDebugDraw.h
${ZERO_ROOT_DIR}/External/recast/include/DetourLocalBoundary.h
${ZERO_ROOT_DIR}/External/recast/include/DetourMath.h
${ZERO_ROOT_DIR}/External/recast/include/DetourNavMesh.h
${ZERO_ROOT_DIR}/External/recast/include/DetourNavMeshBuilder.h
${ZERO_ROOT_DIR}/External/recast/include/DetourNavMeshQuery.h
${ZERO_ROOT_DIR}/External/recast/include/DetourNode.h
${ZERO_ROOT_DIR}/External/recast/include/DetourObstacleAvoidance.h
${ZERO_ROOT_DIR}/External/recast/include/DetourPathCorridor.h
${ZERO_ROOT_DIR}/External/recast/include/DetourPathQueue.h
${ZERO_ROOT_DIR}/External/recast/include/DetourProximityGrid.h
${ZERO_ROOT_DIR}/External/recast/include/DetourStatus.h
${ZERO_ROOT_DIR}/External/recast/src/DetourCrowd.cpp
${ZERO_ROOT_DIR}/External/recast/src/DetourLocalBoundary.cpp
${ZERO_ROOT_DIR}/External/recast/src/DetourObstacleAvoidance.cpp
${ZERO_ROOT_DIR}/External/recast/src/DetourPathCorridor.cpp
${ZERO_ROOT_DIR}/External/recast/src/DetourPathQueue.cpp
${ZERO_ROOT_DIR}/External/recast/src/DetourProximityGrid.cpp
)

target_link_directories(navigation
  PUBLIC
  ${ZERO_ROOT_DIR}/External/recast/lib/win64_vs2017/release
)

target_sources(navigation
  PRIVATE
    ${navigation_src}
    ${CMAKE_CURRENT_LIST_DIR}/NavigationStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/NavigationStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp

)

target_link_libraries(navigation
  PUBLIC
    Common
    Zilch
    core
    engine
    renderer
    recast
)

if(BUILD_EDITOR)
  target_link_libraries(navigation
    PUBLIC
      editor
  )
endif()

target_include_directories(navigation
  PUBLIC
  ${ZERO_ROOT_DIR}/External/recast/include
)
