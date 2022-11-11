include_directories(.)

find_package(CUDAToolkit REQUIRED)

add_library(matmat 
    matmat.hpp
    mpi_matmat.cpp
    cuda_matmat.cpp
)

target_link_libraries(matmat ${MPI_LIBRARIES} CUDA::cudart ${EXTERNAL_LIBS})

include_directories(".")
if(ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
endif()
