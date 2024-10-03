package org.argeo.jjml.llama;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import org.argeo.jjml.ggml.GgmlNumaStrategy;

/** Wrapper to the llama.cpp backend. There can be only one instance. */
public class LlamaCppBackend {
	public final static String SYSTEM_PROPERTY_LIBPATH_LLAMACPP = "org.argeo.jjml.libpath.llamacpp";
	public final static String SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA = "org.argeo.jjml.libpath.jjml.llama";

	private static boolean initialized = false;
	private static boolean librariesLoaded = false;

	private static Path llamaLibraryPath;
	private static Path jjmlLlamaLibraryPath;

	/*
	 * NATIVE
	 */
	static native void doInit();

	static native void doNumaInit(int numaStrategyCode);

	static native void doDestroy();

	/*
	 * LIFECYCLE
	 */
	public static void init() {
		init(null);
	}

	public static void init(GgmlNumaStrategy numaStrategy) {
		if (initialized)
			throw new IllegalStateException("llama.cpp backend is already initialized.");

		// Initialization fails if explicitly found libraries are not found, since it
		// assumed that it will be used in a development or debugging context, which can
		// be messy.
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
		if (llamaLibraryPath != null)
			System.load(llamaLibraryPath.toAbsolutePath().toString());
		if (jjmlLlamaLibraryPath != null)
			System.load(jjmlLlamaLibraryPath.toAbsolutePath().toString());
		System.loadLibrary("org_argeo_jjml_llama");
		librariesLoaded = true;

		doInit();
		initialized = true;
		if (numaStrategy != null)
			doNumaInit(numaStrategy.getCode());
	}

	public static void destroy() {
		doDestroy();
		initialized = false;
		// TODO classloading so that libraries can be unloaded by garbage collection
	}

	public static void setJjmlLlamaLibraryPath(Path jjmlLlamaLibraryPath) {
		LlamaCppBackend.jjmlLlamaLibraryPath = jjmlLlamaLibraryPath;
	}

	public static void setLlamaLibraryPath(Path llamaLibraryPath) {
		LlamaCppBackend.llamaLibraryPath = llamaLibraryPath;
	}
}
