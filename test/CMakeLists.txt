MESSAGE(STATUS "Test CXX Flags: ${CMAKE_CXX_FLAGS}")

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} gtest.a pthread glog gtest_main.a OpenMP::OpenMP_CXX X11 lua5.1)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    add_dependencies(${TESTNAME} ${catkin_EXPORTED_TARGETS})
endmacro()

if (CATKIN_ENABLED_TESTING)
  catkin_add_gtest(solver_test solver_test.cc)
  target_link_libraries(solver_test ${catkin_LIBRARIES})
  add_dependencies(solver_test ${catkin_EXPORTED_TARGETS})
else()
  # build the packages using the above macro.
  package_add_test(solver_test solver_test.cc)
endif()
