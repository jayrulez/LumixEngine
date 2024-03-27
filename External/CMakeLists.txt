add_compile_options(${ZERO_C_CXX_EXTERNAL_FLAGS})

# OS dependent static/source libraries.
add_subdirectory(Curl)
add_subdirectory(GL)
if(NOT ${ZERO_PLATFORM} STREQUAL "Windows")
  add_subdirectory(SDL)
endif()

# Cross platform source libraries.
add_subdirectory(ZeroFreeType)
add_subdirectory(Glew)
add_subdirectory(Libgit2)
add_subdirectory(Libpng)
add_subdirectory(Nvtt)
add_subdirectory(Opus)
add_subdirectory(ZLib)
