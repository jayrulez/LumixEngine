add_library(core)

zero_setup_library(core ${CMAKE_CURRENT_LIST_DIR} TRUE)
zero_use_precompiled_header(core ${CMAKE_CURRENT_LIST_DIR})

set(core_src
    ${CMAKE_CURRENT_LIST_DIR}/allocator.hpp
    ${CMAKE_CURRENT_LIST_DIR}/allocators.cpp
    ${CMAKE_CURRENT_LIST_DIR}/allocators.hpp
    ${CMAKE_CURRENT_LIST_DIR}/array.hpp
    ${CMAKE_CURRENT_LIST_DIR}/associative_array.hpp
    ${CMAKE_CURRENT_LIST_DIR}/atomic.hpp
    ${CMAKE_CURRENT_LIST_DIR}/color.hpp
    ${CMAKE_CURRENT_LIST_DIR}/command_line_parser.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core.hpp
    ${CMAKE_CURRENT_LIST_DIR}/crt.hpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/delegate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/delegate_list.hpp
    ${CMAKE_CURRENT_LIST_DIR}/fibers.hpp
    ${CMAKE_CURRENT_LIST_DIR}/geometry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/geometry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hash.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hash.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hash_map.hpp
    ${CMAKE_CURRENT_LIST_DIR}/job_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/job_system.hpp
    ${CMAKE_CURRENT_LIST_DIR}/log.cpp
    ${CMAKE_CURRENT_LIST_DIR}/log.hpp
    ${CMAKE_CURRENT_LIST_DIR}/math.cpp
    ${CMAKE_CURRENT_LIST_DIR}/math.hpp
    ${CMAKE_CURRENT_LIST_DIR}/metaprogramming.hpp
    ${CMAKE_CURRENT_LIST_DIR}/os.hpp
    ${CMAKE_CURRENT_LIST_DIR}/page_allocator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/page_allocator.hpp
    ${CMAKE_CURRENT_LIST_DIR}/path.cpp
    ${CMAKE_CURRENT_LIST_DIR}/path.hpp
    ${CMAKE_CURRENT_LIST_DIR}/profiler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/profiler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/queue.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ring_buffer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/simd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/span.hpp
    ${CMAKE_CURRENT_LIST_DIR}/stack_array.hpp
    ${CMAKE_CURRENT_LIST_DIR}/stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stream.hpp
    ${CMAKE_CURRENT_LIST_DIR}/string.cpp
    ${CMAKE_CURRENT_LIST_DIR}/string.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sync.hpp
    ${CMAKE_CURRENT_LIST_DIR}/thread.hpp
)

if (${ZERO_PLATFORM} STREQUAL "Windows")
  set(core_src ${core_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/atomic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/fibers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/os.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/simple_win.hpp
    ${CMAKE_CURRENT_LIST_DIR}/win/sync.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/thread.cpp
  
  )
endif()

if (${ZERO_PLATFORM} STREQUAL "Linux")
  set(core_src ${core_src}
    ${CMAKE_CURRENT_LIST_DIR}/linux/atomic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/fibers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/os.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/sync.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/thread.cpp
  )
endif()

target_sources(core
  PRIVATE
  	${core_src}

    ${CMAKE_CURRENT_LIST_DIR}/CoreStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CoreStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp
)

target_link_libraries(core
  PUBLIC
    Common
    Geometry
    Meta
    Platform
    Serialization
    SpatialPartition
    Zilch
)

target_include_directories(core
  PRIVATE
    ${ZERO_ROOT_DIR}/External
)
