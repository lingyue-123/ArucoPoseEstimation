SDK_INC = ../CrobotpOSSDK/cpp/include
SDK_LIB = ../CrobotpOSSDK/bin

.PHONY: all clean

all: robot_bridge.so

robot_bridge.so: robot_bridge.cpp
	g++ -std=c++17 -shared -fPIC -o $@ $< \
		-I$(SDK_INC) \
		-L$(SDK_LIB) -lRobotService \
		-Wl,-rpath,$(abspath $(SDK_LIB))

clean:
	rm -f robot_bridge.so
