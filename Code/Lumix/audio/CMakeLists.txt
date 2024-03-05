add_library(audio)

welder_setup_library(audio ${CMAKE_CURRENT_LIST_DIR} TRUE)
welder_use_precompiled_header(audio ${CMAKE_CURRENT_LIST_DIR})

set(audio_src
    ${CMAKE_CURRENT_LIST_DIR}/audio_device.h
    ${CMAKE_CURRENT_LIST_DIR}/audio_module.cpp
    ${CMAKE_CURRENT_LIST_DIR}/audio_module.h
    ${CMAKE_CURRENT_LIST_DIR}/audio_system.cpp
    ${CMAKE_CURRENT_LIST_DIR}/audio_system.h
    ${CMAKE_CURRENT_LIST_DIR}/clip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/clip.h
)

if (${WELDER_PLATFORM} STREQUAL "Windows")
  set(audio_src ${audio_src}
    ${CMAKE_CURRENT_LIST_DIR}/win/audio_device.cpp
  )
endif()

if (${WELDER_PLATFORM} STREQUAL "Linux")
  set(audio_src ${audio_src}
    ${CMAKE_CURRENT_LIST_DIR}/linux/audio_device.cpp
  )
endif()

if(BUILD_EDITOR)
  set(audio_src ${audio_src}
    ${CMAKE_CURRENT_LIST_DIR}/editor/audio_plugins.cpp
  )
endif()

set(audio_src ${audio_src}
${WELDER_ROOT_DIR}/External/stb/stb_vorbis.cpp
)

target_sources(audio
  PRIVATE
    ${audio_src}
    ${CMAKE_CURRENT_LIST_DIR}/AudioStandard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/AudioStandard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Precompiled.hpp
)

target_link_libraries(audio
  PUBLIC
    Common
    Zilch
    foundation
	  engine
    dxguid
)

if(BUILD_EDITOR)
  target_link_libraries(audio
    PUBLIC
      editor
  )
endif()
