#!/bin/sh

A2_BASE=../../output/a2

LD_LIBRARY_PATH=$A2_BASE/lib/local \
java -ea \
 -Djava.library.path=$A2_BASE/lib/local \
 --module-path $A2_BASE/lib/org.argeo.jjml \
 --add-modules org.argeo.jjml \
 java/examples/JjmlSmokeTests.java \
 allenai/OLMo-2-0425-1B-Instruct-GGUF
