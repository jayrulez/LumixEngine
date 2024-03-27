add_subdirectory(Foundation)
add_definitions(-D_HAS_EXCEPTIONS=0)
remove_definitions(-DUNICODE -D_UNICODE)
add_subdirectory(Lumix)
