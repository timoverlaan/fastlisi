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

nanobind_add_module(_fastlisi
    NB_STATIC
    src/bindings.cpp
)
target_include_directories(_fastlisi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- OpenMP (optional) -------------------------------------------------------
# The neighbor search is parallelized over cells. Everything is guarded by
# #ifdef _OPENMP, so a toolchain without OpenMP still builds and simply runs
# single-threaded.
find_package(OpenMP)

# AppleClang does not find Homebrew's libomp on its own.
if(NOT OpenMP_CXX_FOUND AND APPLE)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/include/omp.h")
        message(STATUS "Using Homebrew libomp at ${LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        find_package(OpenMP)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled: LISI will run multi-threaded")
    target_link_libraries(_fastlisi PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found: LISI will run single-threaded")
endif()

install(TARGETS _fastlisi LIBRARY DESTINATION fastlisi)
