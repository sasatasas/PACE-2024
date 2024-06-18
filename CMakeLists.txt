cmake_minimum_required(VERSION 3.18)
project(MILP VERSION 1.0 LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  or-tools
  GIT_REPOSITORY https://github.com/google/or-tools.git
  GIT_TAG        main
)
set(BUILD_DEPS ON)
set(BUILD_DOC OFF)
set(BUILD_SAMPLES OFF)
set(BUILD_EXAMPLES OFF)
# option(BUILD_SHARED_LIBS "Build shared libraries (.dll)." OFF)
#   set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
#   set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
#   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS "-static")

# After the following call, the CMake targets defined by OR-Tools
# will be defined and available to the rest of the build
FetchContent_MakeAvailable(or-tools)

add_executable(milp_pace milp.cpp)
target_link_libraries(milp_pace ortools::ortools)