package org.argeo.jjml.llama;

import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

/**
 * Loads the shared libraries, possibly using system properties explicitly
 * setting them. If it is used to set them programmatically, it must be done
 * before accessing any other class.
 */
public class LlamaCppNative {
	/**
	 * System property to explicitly specify the path of the GGML shared library to
	 * use.
	 */
	public final static String SYSTEM_PROPERTY_LIBPATH_GGML = "jjml.libpath.ggml";
	/**
	 * System property to explicitly specify the path of the llama.cpp shared
	 * library to use.
	 */
	public final static String SYSTEM_PROPERTY_LIBPATH_LLAMACPP = "jjml.libpath.llamacpp";
	/**
	 * System property to explicitly specify the path of the JNI shared library to
	 * use.
	 */
	public final static String SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA = "jjml.libpath.jjml.llama";

	/**
	 * Environment properties enabling to swap NVRAM to physical memory if needed.
	 * 
	 * @see <a href=
	 *      "https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md">llama.cpp
	 *      build documentation</a>
	 */
	public final static String ENV_GGML_CUDA_ENABLE_UNIFIED_MEMORY = "GGML_CUDA_ENABLE_UNIFIED_MEMORY";

	private final static String JJML_LAMA_LIBRARY_NAME = "Java_"
			+ LlamaCppNative.class.getPackageName().replace('.', '_');

	private final static Logger logger = System.getLogger(LlamaCppNative.class.getName());

	private static boolean librariesLoaded = false;

	private static Path ggmlLibraryPath;
	private static Path llamaLibraryPath;
	private static Path jjmlLlamaLibraryPath;

	/*
	 * STATIC UTILITIES
	 */

	static void ensureLibrariesLoaded() {
		if (librariesLoaded)
			return;
		loadLibraries();
	}

	static void loadLibraries() {
		checkLibrariesNotLoaded();

		// Initialization fails if explicitly found libraries are not found, since it
		// assumed that it will be used in a development or debugging context, which can
		// be messy.
		Optional.ofNullable(System.getProperty(SYSTEM_PROPERTY_LIBPATH_GGML)).ifPresent((path) -> {
			ggmlLibraryPath = Paths.get(path);
			if (!Files.exists(ggmlLibraryPath))
				throw new IllegalArgumentException(
						SYSTEM_PROPERTY_LIBPATH_GGML + " " + ggmlLibraryPath + " does not exist");
		});
		Optional.ofNullable(System.getProperty(SYSTEM_PROPERTY_LIBPATH_LLAMACPP)).ifPresent((path) -> {
			llamaLibraryPath = Paths.get(path);
			if (!Files.exists(llamaLibraryPath))
				throw new IllegalArgumentException(
						SYSTEM_PROPERTY_LIBPATH_LLAMACPP + " " + llamaLibraryPath + " does not exist");
		});
		Optional.ofNullable(System.getProperty(SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA)).ifPresent((path) -> {
			jjmlLlamaLibraryPath = Paths.get(path);
			if (!Files.exists(jjmlLlamaLibraryPath))
				throw new IllegalArgumentException(
						SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA + " " + jjmlLlamaLibraryPath + " does not exist");
		});

		if (ggmlLibraryPath != null) {
			System.load(ggmlLibraryPath.toAbsolutePath().toString());
			logger.log(Level.WARNING, "GGML library loaded from " + ggmlLibraryPath);
		}
		if (llamaLibraryPath != null) {
			System.load(llamaLibraryPath.toAbsolutePath().toString());
			logger.log(Level.WARNING, "llama.cpp library loaded from " + llamaLibraryPath);
		}

		if (jjmlLlamaLibraryPath != null) {
			System.load(jjmlLlamaLibraryPath.toAbsolutePath().toString());
		} else {
			// default behavior
			System.loadLibrary(JJML_LAMA_LIBRARY_NAME);
		}

		// TODO register the stack where libraries were already loaded
		librariesLoaded = true;
	}

	public static void setJjmlLlamaLibraryPath(Path jjmlLlamaLibraryPath) {
		checkLibrariesNotLoaded();
		LlamaCppNative.jjmlLlamaLibraryPath = jjmlLlamaLibraryPath;
	}

	public static void setLlamaLibraryPath(Path llamaLibraryPath) {
		checkLibrariesNotLoaded();
		LlamaCppNative.llamaLibraryPath = llamaLibraryPath;
	}

	public static void setGgmlLibraryPath(Path ggmlLibraryPath) {
		checkLibrariesNotLoaded();
		LlamaCppNative.ggmlLibraryPath = ggmlLibraryPath;
	}

	/** Fails if libraries already loaded. */
	private static void checkLibrariesNotLoaded() {
		if (librariesLoaded)
			throw new IllegalStateException("Shared libraries are already loaded.");
	}

	/** singleton */
	private LlamaCppNative() {
	}
}
