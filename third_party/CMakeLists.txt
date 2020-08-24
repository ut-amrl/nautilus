add_subdirectory(googletest)
include_directories(${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
link_directories(${PROJECT_SOURCE_DIR}/third_party/googletest/lib)
# Now build the libraries that require GTest
set(CSM_LIBRARY TRUE)
add_subdirectory(csm)
set(VECTOR_MAPS_LIBRARY TRUE)
add_subdirectory(vector_maps)
