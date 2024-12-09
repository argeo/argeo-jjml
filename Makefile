# Convenience Makefile based on default Argeo SDK conventions

export JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF8 # Required on Windows

BUILD_BASE=$(abspath ../output/$(notdir $(CURDIR)))

all:
	mkdir -p $(BUILD_BASE)
	cmake -B $(BUILD_BASE) .
	cmake --build $(BUILD_BASE) -j $(shell nproc);

clean:
	if [ -d $(BUILD_BASE) ]; then cmake --build $(BUILD_BASE) --target clean; fi;

install:
	cmake --build ../output/$(BUILD_BASE) --target install
