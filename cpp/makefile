

PythonLIBS = `pkg-config --cflags --libs python2`
PyUblas = -I/usr/local/lib/python2.7/dist-packages/pyublas/include

all:
	g++ --std=c++0x -Wall -O3 -funroll-loops ${PythonLIBS} ${PyUblas} -DPIC -shared -fPIC -o myTest.so source.cpp -lboost_python

clean:
	rm *.so
	
