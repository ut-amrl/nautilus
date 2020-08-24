add_subdirectory(googletest)
include_directories(googletest/bin)
# Now build the libraries that require GTest
set(CSM_LIBRARY TRUE)
add_subdirectory(csm)
set(VECTOR_MAPS_LIBRARY TRUE)
add_subdirectory(vector_maps)