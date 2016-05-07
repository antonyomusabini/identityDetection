CXX ?= g++

CXXFLAGS += -c -Wall $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags sdl) -I/usr/include/SDL
LDFLAGS += $(shell pkg-config --libs opencv) $(shell pkg-config --libs sdl) -lboost_system -lboost_filesystem -pthread -lSDL_mixer -std=c++11

all: identifyAndSpeak

identifyAndSpeak: identifyAndSpeak.o; $(CXX) $< -o $@ $(LDFLAGS)

%.o: %.cpp; $(CXX) $< -o $@ $(CXXFLAGS)

clean: ; rm -f identifyAndSpeak.o identifyAndSpeak





