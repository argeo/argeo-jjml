Argeo JJML provides low-level Java bindings for the [ggml](https://github.com/ggerganov/ggml) family of machine learning libraries, especially [llama.cpp](https://github.com/ggerganov/llama.cpp).

It is written in C++ and relies solely on the plain ggml-*.so/dll and llama.so/dll libraries and their headers. That is, it does not use llama.cpp's "common" layer but rather provides a subset of its features.

It does not depend on any Argeo or third-party Java library and is built with CMake. No tooling or application is provided, except a basic CLI for testing and development purposes. Focus is on providing robust support for batch and embeddings rather than end-user chat assistants.

This has been extracted from another application layer providing agents based on and integrating with the Argeo platform, as it became clear that it could be generally useful and that we would also benefit from others using it in various settings. **Contrary to our usual policy, we are therefore open to feedback and contributions, and we will do reasonable efforts to fix reported issues.** 

# Features
- Java 11+ support, JPMS and OSGi metadata
- Parallel batches
- Embeddings
- Chat templates
- (De)tokenization in UTF-8 (conversion in Java) or UTF-16 (conversion in C++)
- Combination and configuration of (native) samplers from the Java side
- API for implementing samplers in Java

# Build
The build relies on [argeo-build](https://github.com/argeo/argeo-build) scripts as a git submodule, and reference versions of both [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp) are provided as git submodules as well. One should therefore always use `git pull --recurse-submodules` when updating.

```
git clone --recurse-submodules https://github.com/argeo/argeo-jjml
cd argeo-jjml
cmake -B ../output/argeo-jjml
cmake --build ../output/argeo-jjml
```

If the libraries are found at expected locations (/usr, /usr/local, etc.) they will be used (assuming that the related includes are available as well), otherwise the reference submodules will be built as well.

In order to force building with the reference submodules even if the libraries are locally available, use `-DJJML_FORCE_BUILD_TP=ON`.

The primary target is Linux (especially Debian Stable), but we could already build it on Windows with MSYS2.

# Status
The overall architecture is in place and after a few months following closely [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp) development and going through various API changes on their side, we are confident that the Java API will be able to stay reasonably stable once a stable version is released.

The goal is to release a stable version relatively soon (target is Q2 2025).

Planned tasks:
- Shift/rewind context
- Improve build and development environment (feedback welcome!)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) integration
- Android integration (low priority)

# License
Argeo JJML is dual-licensed to the general public with the LGPL v2.1 (or later version) license or the EPL v2 license.

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

# Alternatives for using machine learning with Java
- [java-llama.cpp](https://github.com/kherud/java-llama.cpp) - The Java bindings referenced by the llama.cpp project. It relies on llama.cpp "common" layer and strives to provide the `llama-server` features set. It should therefore be more complete in terms of features while slightly more heavyweight. Argeo JJML provides a different approach, not a competing one.
- [Jlama](https://github.com/tjake/Jlama) - An inference engine written in Java and based on the latest advancement in Java techhnology (esp. the new Vector API). Supports models in *.safetensors format but (at the time of writing) not in GGUF format.
- [llama3.java](https://github.com/mukel/llama3.java) - A very short plain Java implementation based on the new Vector API. Supports only Meta's llama 3.x models (in GGUF format).
- [langchain4j](https://github.com/langchain4j/langchain4j) - A comprehensive LLM framework in Java with various backends, including [ollama](https://github.com/ollama/ollama) (and therefore llama.cpp).
