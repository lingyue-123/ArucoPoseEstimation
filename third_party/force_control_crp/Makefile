SDK_ROOT ?= /home/nvidia/CrobotpOSSDK-1.3.4-Linux-armv8-ubuntu20
SDK_INC := $(SDK_ROOT)/cpp/include

TARGET := libforcecontrol_crp.so
SRC := src/it_robotposTEST8.cpp src/ForceControl.c src/IdentifyCommon.c src/OnLineFit.c

CXXFLAGS := -std=c++17 -shared -fPIC -O2 -I$(SDK_INC) -Isrc
LDFLAGS := -L. -lRobotService -lpthread -ldl -lm -Wl,-rpath,'$$ORIGIN'

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	g++ $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
