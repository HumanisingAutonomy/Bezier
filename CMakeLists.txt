cmake_minimum_required(VERSION 3.12.4)
project(bezier
  LANGUAGES CXX
  VERSION 0.3.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

set(Bezier_SRC
  ${PROJECT_SOURCE_DIR}/src/bezier.cpp
  ${PROJECT_SOURCE_DIR}/src/polycurve.cpp
  )

set(Bezier_INC
  ${PROJECT_SOURCE_DIR}/include/Bezier/declarations.h
  ${PROJECT_SOURCE_DIR}/include/Bezier/bezier.h
  ${PROJECT_SOURCE_DIR}/include/Bezier/polycurve.h
  )


# detect if we have been included via add_subdirectory
set(NOT_IN_SUB_PROJECT OFF)
if (PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
  set(NOT_IN_SUB_PROJECT ON)
endif()

# Options
option(BUILD_SHARED_LIBS "Build shared library (.so) instead of static one (.a)" OFF)
# only include tests by default if we are building this project directly
option(BEZIER_BUILD_TESTS "Build tests" ${NOT_IN_SUB_PROJECT})

message(STATUS "Bezier bulding tests: ${BEZIER_BUILD_TESTS}")
# build rules

if(BUILD_SHARED_LIBS)
  add_library(bezier SHARED ${Bezier_SRC})
else()
  add_library(bezier STATIC ${Bezier_SRC})
endif()


target_include_directories(bezier PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(bezier PUBLIC Eigen3::Eigen)
set_property(TARGET bezier PROPERTY POSITION_INDEPENDENT_CODE ON)

set_target_properties(bezier PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(bezier PROPERTIES PUBLIC_HEADER "${Bezier_INC}")

# install rules
install(TARGETS bezier
  EXPORT bezier-export DESTINATION "lib"
  PUBLIC_HEADER DESTINATION "include/Bezier")
install(EXPORT bezier-export DESTINATION "lib/cmake/Bezier" FILE BezierConfig.cmake)

add_library(bezier::bezier ALIAS bezier)

if(BEZIER_BUILD_TESTS)
  add_subdirectory(tests)
endif()