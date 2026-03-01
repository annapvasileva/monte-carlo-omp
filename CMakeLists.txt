cmake_minimum_required(VERSION 3.12)

project(
    labwork4
    VERSION 1.0
    DESCRIPTION "Hardware of computing systems Year 2024 labwork 4"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O2")
       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O2")
    ENDIF()
ENDIF()

add_executable(${PROJECT_NAME} main.cpp)
add_subdirectory(include)
add_library(hit_lib hit.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE hit_lib pcg_lib)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})