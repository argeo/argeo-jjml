# Enterprise-grade Java bindings for the ggml ecosystem ###

## Generative AI locally, integrated into existing Java systems ##
Argeo JJML provides low-level Java bindings for the [ggml](https://github.com/ggml-org/ggml) family of machine learning libraries, especially [llama.cpp](https://github.com/ggml-org/llama.cpp) which allows to run locally open-weights large language models (LLMs, aka. "generative AI").

The main goal of this lightweight component is to provide an enterprise-grade quality mechanism to integrate local LLMs into existing Java systems, with stable Java APIs, a small auditable code base, and essentially no impact on other components.

While the field of LLMs is moving very fast, with new open-weight models being published on a monthly basis, there is already a lot that can be done reliably, and the ggml and llama.cpp projects have proven that they can combine a vibrant community of contributors with good software engineering. Argeo JJML provides a kind of "shock absorber" for the Java ecosystem, smoothing the unavoidable native API breakages, supporting old Java versions, and avoiding the deployment of Python-based solutions in an enterprise setting, when only inference is needed.

The native interface layer is written in C++ and relies solely on the plain ggml-*.so/dll and llama.so/dll shared libraries and their headers. That is, it does not use llama.cpp's "common" layer, but rather provides a subset of its features.

The Java layer does not depend on any Argeo or third-party Java libraries, and is also built with CMake. It has no other dependency than the `java.base` module of the standard Java runtime, and is therefore well-suited for creating stripped-down Java runtimes with the `jlink` utility.

No tooling or application is provided, except some examples for testing and development purposes. Focus is on stability rather than supporting the latest features. Usable features such as chatbots, RAG, HTTP APIs, etc. should be implemented on top of this component, typically using third-party Java libraries and frameworks.

The applications targeted by this library are mostly enterprise Java systems in regulated, sovereign or sustainable industries (where running LLMs on premise or on a private cloud is a requirement). But it can also be helpful when developing or patching ggml-based native libraries, by providing a simple way to write robust scripted tests or small prototypes using modern Java features such as `jdk.jshell`, `jdk.httpserver`, WebSocket client, etc.

## Features ##
- Java 11, 17, 21 and 25 support
- Persistence of context state, typically in order to "pre-compile" prompt prefixes
- Parallel batches
- Embeddings
- Chat templates (limited to those embedded in llama.cpp)
- API-less user/assistant dialog based on standard `java.util.function` interfaces
- Combination and configuration of the native samplers from the Java side
- API for implementing samplers in pure Java
- JPMS and OSGi metadata
- Android support (from SDK version 26, example project in the `unstable` branch)

## Build ##
The build relies only on CMake and the [argeo-build](https://github.com/argeo/argeo-build) scripts (as a git submodule). Pinned reference versions of both [ggml](https://github.com/ggml-org/ggml) and [llama.cpp](https://github.com/ggml-org/llama.cpp) are provided as git submodules as well. *One should therefore always use `git pull --recurse-submodules` when updating.*

```
sudo apt install default-jdk # install Java
# sudo apt install libllama-dev # llama.cpp dev packages, where available

git clone --recurse-submodules https://github.com/argeo/argeo-jjml
cd argeo-jjml
cmake -B build/default -DJAVA_HOME=/usr/lib/jvm/default-java
cmake --build build/default -j $(nproc)
```

The built artifacts are located under `<CMake build dir>/../a2`.

One can then run some smoke tests:
```
java -ea \
 -cp "build/a2/org.argeo.jjml/*" \
 -Djava.library.path=build/a2/lib/$(uname -m)-linux-gnu/org.argeo.jjml \
 sdk/jbin/JjmlSmokeTests.java \
 allenai/OLMo-2-0425-1B-Instruct-GGUF
```
or a basic CLI:
```
java \
 -cp "build/a2/org.argeo.jjml/*" \
 -Djava.library.path=build/a2/lib/$(uname -m)-linux-gnu/org.argeo.jjml \
 sdk/jbin/JjmlDummyCli.java \
 allenai/OLMo-2-0425-1B-Instruct-GGUF
```

If the shared libraries are found at the usual locations (`/usr`, `/usr/local`, etc., as well as Debian-specific `/usr/lib/\*/ggml` and `/usr/lib/\*/llama`) they will be used, then assuming that the related includes, cmake-* configs, etc. are available as well. Otherwise, the reference ggml and llama.cpp submodules will be built in addition to the Java bindings.

If the ggml and llama.cpp libraries are rebuilt, all the `GGML_*` and `LLAMA_*` CMake options are available to their respective builds, which can therefore be customized exactly like regular llama.cpp builds. In order to add them when testing, extend the JNI path, using `-Djava.library.path=build/a2/lib/$(uname -m)-linux-gnu/org.argeo.jjml:build/a2/lib/$(uname -m)-linux-gnu/org.argeo.tp.ggml`.

In order to force building with the reference submodules even if the libraries are locally available, use `-DJJML_FORCE_BUILD_TP=ON` when configuring CMake.

Reciprocally, use `-DJJML_DO_NOT_BUILD_TP=ON` in order to make sure that the build is using the system libraries for ggml and llama.cpp (and not automatically defaulting to build the submodules).

When building the reference submodules, setting `-DJJML_FORCE_BUILD_LLAMA_GGML=ON` will build with the ggml version included in `native/tp/llama.cpp`. The default is to build with the separate `native/tp/ggml` reference submodule. This is useful when testing with the latest version of llama.cpp or a development branch.

By default, the Java part of JJML is built for the minimal supported Java version (that is, Java 11). In order to build for a later Java version, set `A2_JAVA_RELEASE` when configuring the CMake build, e.g. `-DA2_JAVA_RELEASE=21`.

While a lot of work goes into making this build straightforward and portable, there must be a clear baseline:
- The reference build for Linux is on Debian Sid (amd64/x86_64 and arm64/aarch64), using the official Debian packages for ggml and llama.cpp. (JJML's lead developer is a regular contributor to this Debian packaging effort)
- The reference build for Windows is with the Microsoft MSVC compiler. (see example below)
- Other operating systems (notably MacOS, Android) could work but are not currently considered
When reporting build issues on a given platform, please first check whether a reference build is working.

An example Windows build would be (in a PowerShell terminal):

```
winget install Microsoft.OpenJDK.21 # install MS JDK
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
 -B build\default `
 -DJAVA_HOME="C:/Program Files/Microsoft/jdk-21.0.8.9-hotspot" # Note regular slashes
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
 --build build\default
```

## Status ##
Argeo JJML is currently in open beta, the last phase before a first stable release.

All features of the future stable release are implemented and should not change significantly. Work has already started on commercial projects using it in various industries.

Now is a good time to start using it in a given context, as we can quickly fix issues or integrate feedback, while the APIs already have taken their shape.

Future features:
- Shift/rewind/fork context
- Speech recognition and transcription with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) integration (already working in the `unstable` branch)
- Image recognition and multimodal support with llama.cpp's [mtmd](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd) (work-in-progress in the `unstable` branch)

## Contact ##
Issues can be raised via [GitHub](https://github.com/argeo/argeo-jjml/issues) or Debian's [Salsa](https://salsa.debian.org/mbaudier/libjjml-java/-/issues) (especially when related to packaging or portability).

All other queries, typically new features, should be directed to Mathieu Baudier via [LinkedIn](https://www.linkedin.com/in/mbaudier/). Argeo GmbH also provide support and consulting services around the integration of JJML into existing Java systems.

## Licensing ##
Argeo JJML is dual-licensed:
- LGPL v2.1 (or later version)
- EPL v2, with GPL as a possible secondary license

```
Copyright 2024-2025 Mathieu Baudier

Copyright 2024-2025 Argeo GmbH

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; if not, see <https://www.gnu.org/licenses>.

## Alternative licenses

As an alternative, this Program is also provided to you under the terms and 
conditions of the Eclipse Public License version 2.0 or any later version. 
A copy of the Eclipse Public License version 2.0 is available at 
http://www.eclipse.org/legal/epl-2.0.

This Source Code may also be made available under the following 
Secondary Licenses when the conditions for such availability set forth 
in the Eclipse Public License, v. 2.0 are satisfied: 
GNU General Public License, version 2.0, or any later versions of that license, 
with additional EPL and JCR permissions (these additional permissions being 
detailed hereafter).
```

See [NOTICE](NOTICE) for more details.

```
SPDX-License-Identifier: LGPL-2.1-or-later OR EPL-2.0 OR LicenseRef-argeo2-GPL-2.0-or-later-with-EPL-and-Apache-and-JCR-permissions
```

## Alternatives for deploying machine learning with Java ##
- [java-llama.cpp](https://github.com/kherud/java-llama.cpp) - The Java bindings referenced by the llama.cpp project. It relies on llama.cpp "common" layer and strives to provide the `llama-server` features set. It should therefore be more complete in terms of features, while slightly more heavyweight. Argeo JJML provides a different approach, not a competing one.
- [Jlama](https://github.com/tjake/Jlama) - An inference engine written in Java and based on the latest advancements in Java technology (esp. the new Vector API). Supports models in *.safetensors format but (at the time of writing) not in GGUF format.
- [llama3.java](https://github.com/mukel/llama3.java) - A very short plain Java implementation based on the new Vector API. Supports only Meta's llama 3.x models (in GGUF format).
- [langchain4j](https://github.com/langchain4j/langchain4j) - A comprehensive LLM framework in Java with various backends, including [ollama](https://github.com/ollama/ollama) (and therefore llama.cpp).
