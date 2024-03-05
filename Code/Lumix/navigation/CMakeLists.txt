add_library(navigation)

welder_setup_library(navigation ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(navigation ${CMAKE_CURRENT_LIST_DIR})

set(navigation_src
    ${CMAKE_CURRENT_LIST_DIR}/navigation_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/navigation_module.h
    ${CMAKE_CURRENT_LIST_DIR}/navigation_system.cpp
)

if(BUILD_EDITOR)
  set(navigation_src ${navigation_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/navigation_plugins.cpp
  )
endif()

set(navigation_src ${navigation_src}
${WELDER_ROOT_DIR}/External/recast/include/DetourAlloc.h
${WELDER_ROOT_DIR}/External/recast/include/DetourAssert.h
${WELDER_ROOT_DIR}/External/recast/include/DetourCommon.h
${WELDER_ROOT_DIR}/External/recast/include/DetourCrowd.h
${WELDER_ROOT_DIR}/External/recast/include/DetourDebugDraw.h
${WELDER_ROOT_DIR}/External/recast/include/DetourLocalBoundary.h
${WELDER_ROOT_DIR}/External/recast/include/DetourMath.h
${WELDER_ROOT_DIR}/External/recast/include/DetourNavMesh.h
${WELDER_ROOT_DIR}/External/recast/include/DetourNavMeshBuilder.h
${WELDER_ROOT_DIR}/External/recast/include/DetourNavMeshQuery.h
${WELDER_ROOT_DIR}/External/recast/include/DetourNode.h
${WELDER_ROOT_DIR}/External/recast/include/DetourObstacleAvoidance.h
${WELDER_ROOT_DIR}/External/recast/include/DetourPathCorridor.h
${WELDER_ROOT_DIR}/External/recast/include/DetourPathQueue.h
${WELDER_ROOT_DIR}/External/recast/include/DetourProximityGrid.h
${WELDER_ROOT_DIR}/External/recast/include/DetourStatus.h
${WELDER_ROOT_DIR}/External/recast/src/DetourCrowd.cpp
${WELDER_ROOT_DIR}/External/recast/src/DetourLocalBoundary.cpp
${WELDER_ROOT_DIR}/External/recast/src/DetourObstacleAvoidance.cpp
${WELDER_ROOT_DIR}/External/recast/src/DetourPathCorridor.cpp
${WELDER_ROOT_DIR}/External/recast/src/DetourPathQueue.cpp
${WELDER_ROOT_DIR}/External/recast/src/DetourProximityGrid.cpp
)

target_link_directories(navigation
  PUBLIC
  ${WELDER_ROOT_DIR}/External/recast/lib/win64_vs2017/release
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
    foundation
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
  ${WELDER_ROOT_DIR}/External/recast/include
)
