MESSAGE(STATUS "Test CXX Flags: ${CMAKE_CXX_FLAGS}")

set(non_findable_libs gflags glog X11 m lua5.1 pthread csm vector_maps Boost::system stdc++)

macro(add_exec NAME)
  add_executable(${NAME} ${ARGN})
  target_link_libraries(${NAME} gtest.a gtest_main.a 
                                ${CERES_LIBRARIES}
                                ${OpenMP_CXX_LIBRARIES}
                                ${non_findable_libs})
  set_target_properties(${NAME} PROPERTIES FOLDER tests)
  add_dependencies(${NAME} ${catkin_EXPORTED_TARGETS} ${catkin_LIBRARIES})
endmacro()

macro(package_add_test TESTNAME)
  add_executable(${TESTNAME} ${ARGN})
  target_link_libraries(${TESTNAME} gtest.a gtest_main.a 
                                ${CERES_LIBRARIES}
                                ${OpenMP_CXX_LIBRARIES}
                                ${non_findable_libs})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
  add_dependencies(${TESTNAME} ${catkin_EXPORTED_TARGETS} ${catkin_LIBRARIES})
  add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endmacro()

macro(custom_catkin_add_test TESTNAME)
  catkin_add_gtest(${TESTNAME} ${ARGN})
  target_link_libraries(${TESTNAME} ${catkin_LIBRARIES})
  add_dependencies(${TESTNAME} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${TESTNAME} gtest.a gtest_main.a 
                                    ${CERES_LIBRARIES}
                                    ${OpenMP_CXX_LIBRARIES}
                                    ${non_findable_libs})
endmacro()

macro(custom_add_test TESTNAME)
  if (CATKIN_ENABLE_TESTING)
    custom_catkin_add_test(${TESTNAME} ${ARGN})
  else()
    package_add_test(${TESTNAME} ${ARGN})
  endif()
endmacro()


custom_add_test(solver_test solver_test.cc)
