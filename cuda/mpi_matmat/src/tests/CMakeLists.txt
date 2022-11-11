add_executable(test_cuda_matmat test_cuda_matmat.cpp)
add_executable(test_mpi_matmat test_mpi_matmat.cpp)

target_link_libraries(test_cuda_matmat matmat gtest pthread)
target_link_libraries(test_mpi_matmat matmat gtest pthread)

add_test(CudaTest ./test_cuda_matmat 100)
add_test(MPITest mpirun -n 2 ./test_mpi_matmat 100)
