g++ -std=c++11 -c ../lib/vector_utils.cpp -o vector_utils.o
nvcc -std=c++11 -c template.cu -o template.o 
nvcc -std=c++11 template.o vector_utils.o -o out
rm vector_utils.o template.o
