# Enterprise-grade Java bindings for using LLMs on premise

## Use core LLM capabilities locally within existing Java systems
Argeo JJML provides low-level Java bindings for the [ggml](https://github.com/ggml-org/ggml) family of machine learning libraries, especially [llama.cpp](https://github.com/ggml-org/llama.cpp) which allows to run locally open-weights large language models (LLMs, aka. "generative AI").

The main goal of this lightweight component is to provide an enterprise-grade quality mechanism to integrate local LLMs into existing Java systems, with stable Java APIs, a small auditable code base, and essentially no impact on other components.

While the field of LLMs is moving very fast, with new open-weight models being published on a monthly basis, there is already a lot that can be done reliably, and the ggml and llama.cpp projects have proven that they can combine a vibrant community of contributors with good software engineering. Argeo JJML provides a kind of "shock absorber" for the Java ecosystem, smoothing the unavoidable native API breakages, supporting old Java versions, and avoiding the deployment of Python-based solutions in an enterprise setting.

The native interface layer is written in C++ and relies solely on the plain ggml-*.so/dll and llama.so/dll shared libraries and their headers. That is, it does not use llama.cpp's "common" layer, but rather provides a subset of its features.

The Java layer does not depend on any Argeo or third-party Java libraries, and is also built with CMake. It has no other dependency than the `java.base` module of the standard Java runtime, and is therefore well-suited for creating stripped-down Java runtimes with the `jlink` utility.

No tooling or application is provided, except some examples for testing and development purposes. Focus is on stability rather than supporting the latest features. Usable features such as chatbots, RAG, HTTP APIs, etc. should be implemented on top of this component, typically using third-party libraries and frameworks.

## Features
- Java 11+ support
- JPMS and OSGi metadata
- Simple user/assistant dialog based on standard functional interfaces
- Persistence of context state, typically in order to "pre-compile" propmt prefixes
- Parallel batches
- Embeddings
- Chat templates (limited to those embedded in llama.cpp)
- Combination and configuration of (native) samplers from the Java side
- API for implementing samplers in pure Java

## Build
The build relies only on CMake and the [argeo-build](https://github.com/argeo/argeo-build) scripts (as a git submodule). Pinned reference versions of both [ggml](https://github.com/ggml-org/ggml) and [llama.cpp](https://github.com/ggml-org/llama.cpp) are provided as git submodules as well. *One should therefore always use `git pull --recurse-submodules` when updating.*

```
git clone --recurse-submodules https://github.com/argeo/argeo-jjml
cd argeo-jjml
cmake -B ../output/argeo-jjml
cmake --build ../output/argeo-jjml
```

If the shared libraries are found at the usual locations (/usr, /usr/local, etc., as well as Debian's /usr/lib/\*/ggml and /usr/lib/\*/llama) they will be used, then assuming that the related includes, camke configs, etc. are available as well. Otherwise, the referenced submodules will be built additionally to the Java bindings.

In order to force building with the reference submodules even if the libraries are locally available, use `-DJJML_FORCE_BUILD_TP=ON` when configuring CMake.

## Status
Argeo JJML is currently in open beta, the last phase before a first stable release.

All features of the future stable release are implemented and should not change significantly. Work has already started on commercial projects using it in various industries.

Future features:
- Shift/rewind context
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) integration
- Android integration (low priority)

## Contact, bug reports, commercial support
All queries should be directed to Mathieu Baudier via [LinkedIn](https://www.linkedin.com/in/mbaudier/). You can expect properly reported bugs to be fixed free of charge, and additional features to require a fee. We can also provide consulting services in order to help you integrate this capabilities into your existing Java systems.

In line with general Argeo policy, no community support is provided, as all our pro-bono efforts always go to non-commercial upstream projects (in that case, mostly contributions to the Debian packaging of ggml and llama.cpp).

## License
Argeo JJML is dual-licensed:
- LGPL v2.1 (or later version)
- EPL v2, with GPL as a possible secondary license

```Copyright 2024-2025 Mathieu Baudier

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

## Alternatives for using machine learning with Java
- [java-llama.cpp](https://github.com/kherud/java-llama.cpp) - The Java bindings referenced by the llama.cpp project. It relies on llama.cpp "common" layer and strives to provide the `llama-server` features set. It should therefore be more complete in terms of features, while slightly more heavyweight. Argeo JJML provides a different approach, not a competing one.
- [Jlama](https://github.com/tjake/Jlama) - An inference engine written in Java and based on the latest advancements in Java technology (esp. the new Vector API). Supports models in *.safetensors format but (at the time of writing) not in GGUF format.
- [llama3.java](https://github.com/mukel/llama3.java) - A very short plain Java implementation based on the new Vector API. Supports only Meta's llama 3.x models (in GGUF format).
- [langchain4j](https://github.com/langchain4j/langchain4j) - A comprehensive LLM framework in Java with various backends, including [ollama](https://github.com/ollama/ollama) (and therefore llama.cpp).
