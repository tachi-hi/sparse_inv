

PythonLIBS = `pkg-config --cflags --libs python2`
PyUblas = -I/usr/local/lib/python2.7/dist-packages/pyublas/include

OBJS=alg.o python.o
all: myTest.so

myTest.so: $(OBJS)
	g++ --std=c++0x -Wall -o $@ -funroll-loops ${PythonLIBS} ${PyUblas} -DPIC -shared -fPIC $(OBJS) -lboost_python

alg.o: alg.cpp
	g++ --std=c++0x -Wall -c -O3 ${PythonLIBS} ${PyUblas} -fPIC $<

python.o: python.cpp
	g++ --std=c++0x -Wall -c -O3 ${PythonLIBS} ${PyUblas} -fPIC $< -lboost_python


clean:
	rm *.so *.o
