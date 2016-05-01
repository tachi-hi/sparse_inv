

PythonLIBS = `pkg-config --cflags --libs python2`
PyUblas = -I/usr/local/lib/python2.7/dist-packages/pyublas/include

all: myTest.so

myTest.so: proc.o source.o
	g++ --std=c++0x -Wall -o $@ -funroll-loops ${PythonLIBS} ${PyUblas} -DPIC -shared -fPIC  proc.o source.o -lboost_python

proc.o: proc.cpp
	g++ --std=c++0x -Wall -c -O3 ${PythonLIBS} ${PyUblas} -fPIC proc.cpp

source.o: source.cpp
	g++ --std=c++0x -Wall -c -O3 ${PythonLIBS} ${PyUblas} -fPIC source.cpp -lboost_python


clean:
	rm *.so *.o
