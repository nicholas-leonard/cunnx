CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)

IF(APPLE)
  # work around for mac os x bug:
  # http://stackoverflow.com/questions/16286588/cuda-5-0-cmake-and-make-failing-on-osx-10-8-3
  if (NOT DEFINED CUDA_HOST_COMPILER AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND EXISTS /usr/bin/gcc)
    set(CUDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by NVCC")
    message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.")
  endif()
ENDIF()

FIND_PACKAGE(CUDA 6.5 REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
    if(CUDA_VERSION VERSION_LESS "8.0")
      MESSAGE(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
    endif(CUDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

# Detect CUDA architecture and get best NVCC flags
IF(COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
  # New CMake function, comes with FindCUDA
  CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA $ENV{TORCH_CUDA_ARCH_LIST})
ELSE()
  # Old way of getting the flags, with extra include file
  INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCudaArch.cmake)
  SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)
ENDIF()
LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})


INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src-cuda init.cu)
FILE(GLOB luasrc *.lua)
SET(luasrc ${luasrc} test/test.lua)
#ADD_TORCH_PACKAGE(cunnx "${src-cuda}" "${luasrc}" "Experiment cuda nn library")

CUDA_ADD_LIBRARY(cunnx MODULE ${src-cuda})
CUDA_ADD_CUBLAS_TO_TARGET(cunnx)
TARGET_LINK_LIBRARIES(cunnx luaT THC TH)
IF(APPLE)
  SET_TARGET_PROPERTIES(cunnx PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
ENDIF()
IF(LUALIB)
  TARGET_LINK_LIBRARIES(cunnx ${LUALIB})
ENDIF()

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(cunnx PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

INSTALL(TARGETS cunnx
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/cunnx")
