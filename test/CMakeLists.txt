find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
#[[ This macro was taken from: https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html]]
macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} ${GTEST_LIBRARIES} pthread)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(SolverTest solver_test.cc)