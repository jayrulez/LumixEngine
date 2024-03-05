add_library(foundation)

welder_setup_library(foundation ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(foundation ${CMAKE_CURRENT_LIST_DIR})

set(foundation_src
    ${CMAKE_CURRENT_LIST_DIR}/allocator.h
    ${CMAKE_CURRENT_LIST_DIR}/allocators.cpp
    ${CMAKE_CURRENT_LIST_DIR}/allocators.h
    ${CMAKE_CURRENT_LIST_DIR}/array.h
    ${CMAKE_CURRENT_LIST_DIR}/associative_array.h
    ${CMAKE_CURRENT_LIST_DIR}/atomic.h
    ${CMAKE_CURRENT_LIST_DIR}/color.h
    ${CMAKE_CURRENT_LIST_DIR}/command_line_parser.h
    ${CMAKE_CURRENT_LIST_DIR}/crt.h
    ${CMAKE_CURRENT_LIST_DIR}/debug.h
    ${CMAKE_CURRENT_LIST_DIR}/delegate.h
    ${CMAKE_CURRENT_LIST_DIR}/delegate_list.h
    ${CMAKE_CURRENT_LIST_DIR}/fibers.h
    ${CMAKE_CURRENT_LIST_DIR}/file_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/file_system.h
    ${CMAKE_CURRENT_LIST_DIR}/foundation.h
    ${CMAKE_CURRENT_LIST_DIR}/geometry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/geometry.h
    ${CMAKE_CURRENT_LIST_DIR}/hash.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hash.h
    ${CMAKE_CURRENT_LIST_DIR}/hash_map.h
    ${CMAKE_CURRENT_LIST_DIR}/job_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/job_system.h
    ${CMAKE_CURRENT_LIST_DIR}/log.cpp
    ${CMAKE_CURRENT_LIST_DIR}/log.h
    ${CMAKE_CURRENT_LIST_DIR}/math.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lmath.h
    ${CMAKE_CURRENT_LIST_DIR}/metaprogramming.h
    ${CMAKE_CURRENT_LIST_DIR}/os.h
    ${CMAKE_CURRENT_LIST_DIR}/page_allocator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/page_allocator.h
    ${CMAKE_CURRENT_LIST_DIR}/path.cpp
    ${CMAKE_CURRENT_LIST_DIR}/path.h
    ${CMAKE_CURRENT_LIST_DIR}/profiler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/profiler.h
    ${CMAKE_CURRENT_LIST_DIR}/queue.h
    ${CMAKE_CURRENT_LIST_DIR}/ring_buffer.h
    ${CMAKE_CURRENT_LIST_DIR}/simd.h
    ${CMAKE_CURRENT_LIST_DIR}/span.h
    ${CMAKE_CURRENT_LIST_DIR}/stack_array.h
    ${CMAKE_CURRENT_LIST_DIR}/stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stream.h
    ${CMAKE_CURRENT_LIST_DIR}/string.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lstring.h
    ${CMAKE_CURRENT_LIST_DIR}/sync.h
    ${CMAKE_CURRENT_LIST_DIR}/thread.h
)

if (${WELDER_PLATFORM} STREQUAL "Windows")
  set(foundation_src ${foundation_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/atomic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/fibers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/os.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/simple_win.h
    ${CMAKE_CURRENT_LIST_DIR}/win/sync.cpp
    ${CMAKE_CURRENT_LIST_DIR}/win/thread.cpp
  
  )
endif()

if (${WELDER_PLATFORM} STREQUAL "Linux")
  set(foundation_src ${foundation_src}
    ${CMAKE_CURRENT_LIST_DIR}/linux/atomic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/fibers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/os.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/sync.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linux/thread.cpp
  )
endif()

target_sources(foundation
  PRIVATE
  	${foundation_src}

    ${CMAKE_CURRENT_LIST_DIR}/FoundationStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FoundationStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp
)

target_link_libraries(foundation
  PUBLIC
    Common
    Geometry
    Meta
    Platform
    Serialization
    SpatialPartition
    Zilch
)

#set_property(TARGET foundation PROPERTY
#  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
#  MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
#)
