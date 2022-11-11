include_directories(${matmat_INCDIR})

add_executable(time_cuda_matmat time_cuda_matmat.cpp)
add_executable(time_mpi_matmat time_mpi_matmat.cpp)

target_link_libraries(time_cuda_matmat matmat)
target_link_libraries(time_mpi_matmat matmat)
