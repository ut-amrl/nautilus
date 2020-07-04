MESSAGE(STATUS "Test CXX Flags: ${CMAKE_CXX_FLAGS}")

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    target_link_libraries(${TESTNAME} gtest gtest_main glog lua5.1 pthread)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(solver_test solver_test.cc)
package_add_test(kdtree_test kdtree_test.cc)
