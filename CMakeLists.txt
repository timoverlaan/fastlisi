cmake_minimum_required(VERSION 3.15...3.30)
project(fastlisi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# Parallelism uses std::thread from the standard library, so there is no
# OpenMP runtime to find, link, or bundle into the wheel.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

nanobind_add_module(_fastlisi
    NB_STATIC
    src/cpp/bindings.cpp
)
target_include_directories(_fastlisi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
target_link_libraries(_fastlisi PRIVATE Threads::Threads)

install(TARGETS _fastlisi LIBRARY DESTINATION fastlisi)
