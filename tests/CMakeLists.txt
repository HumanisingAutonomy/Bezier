
enable_testing()
include(GoogleTest)
include(FetchContent)

FetchContent_Declare(
    GTest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG         v1.15.2
  SYSTEM
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(GTest)

file(GLOB TEST_FILES CONFIGURE_DEPENDS "*.cpp")

add_executable(
    test_bezier
    ${TEST_FILES}
)

target_link_libraries(
    test_bezier
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    bezier::bezier
)
gtest_discover_tests(test_bezier
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_target_properties(test_bezier PROPERTIES FOLDER tests)