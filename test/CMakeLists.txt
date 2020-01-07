MESSAGE(STATUS "Test CXX Flags: ${CMAKE_CXX_FLAGS}")

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} gtest pthread glog gtest_main OpenMP::OpenMP_CXX)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(solver_test solver_test.cc)
package_add_test(correlative_scan_matching_test correlative_scan_matching_test.cc)
