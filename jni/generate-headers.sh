#!/bin/sh
/usr/lib/jvm/default-java/bin/javac \
	-d build/classes \
	../org.argeo.jjml.llama/src/org/argeo/jjml/ggml/*.java
/usr/lib/jvm/default-java/bin/javac -h org_argeo_jjml_llama \
	-cp build/classes \
	-d build/classes \
	../org.argeo.jjml.llama/src/org/argeo/jjml/llama/*.java
